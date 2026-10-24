import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from aspire_eval.client import CloudError, CostCeilingExceeded, OllamaCloudClient
from aspire_eval.config import AppConfig
from aspire_eval.cost import estimate_cost
from aspire_eval.fixtures import TestCase
from aspire_eval.observability import EventLog, write_json
from aspire_eval.parsing import extract_json, message_content
from aspire_eval.prompts import critic_messages, primary_messages
from aspire_eval.schemas import CriticResponse, PrimaryResponse, complete_primary
from aspire_eval.scoring import (
    absolute_misses,
    best_requirement_status,
    reconcile,
    suite_gate_score,
)


class EvaluationPipeline:
    def __init__(
        self,
        config: AppConfig,
        client: OllamaCloudClient,
        events: EventLog,
        output_dir: Path,
        api_key: str,
    ) -> None:
        self.config = config
        self.client = client
        self.events = events
        self.output_dir = output_dir
        self.api_key = api_key
        self.spent_usd = 0.0

    def evaluate(self, run_id: str, tests: list[TestCase]) -> dict:
        run_dir = self.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "input.json", self._input_record(tests))
        primary, primary_usage, primary_error = self._structured_call(
            run_id=run_id,
            stage="primary_judge",
            model_role="primary_judge",
            messages=primary_messages(tests),
            parser=lambda payload: self._parse_primary(payload, tests),
            run_dir=run_dir,
        )
        critic = None
        critic_usage: list[dict] = []
        degraded = False
        if primary is None:
            return self._failed(run_id, tests, primary_usage, primary_error or "Primary judge did not return valid JSON")

        critic, critic_usage, critic_error = self._structured_call(
            run_id=run_id,
            stage="independent_critic",
            model_role="independent_critic",
            messages=critic_messages(tests, primary),
            parser=lambda payload: CriticResponse.model_validate(payload),
            run_dir=run_dir,
        )
        if critic is None:
            degraded = True

        reviews = {item.test_case_id: item for item in critic.reviews} if critic else {}
        reconciled = []
        for test in tests:
            evaluation = next(item for item in primary.evaluations if item.test_case_id == test.id)
            reconciled.append(
                reconcile(
                    test,
                    evaluation,
                    reviews.get(test.id),
                    self.config.pipeline.score_weights,
                )
            )
        best = best_requirement_status(reconciled)
        result = {
            "run_id": run_id,
            "status": "degraded" if degraded else "completed",
            "degraded_reason": critic_error if degraded else None,
            "tests": [self._test_dict(item) for item in reconciled],
            "suite": {
                "gate_score": suite_gate_score(best),
                "mean_quality_score": round(sum(item.quality_score for item in reconciled) / len(reconciled), 2),
                "best_statuses": best,
                "absolute_misses": absolute_misses(best),
            },
            "usage": primary_usage + critic_usage,
            "estimated_cost_usd": round(sum(item["estimated_cost_usd"] or 0 for item in primary_usage + critic_usage), 8),
        }
        write_json(run_dir / "result.json", result)
        self.events.write("run_completed", run_id=run_id, status=result["status"], gate_score=result["suite"]["gate_score"])
        return result

    def _structured_call(self, run_id, stage, model_role, messages, parser, run_dir: Path):
        model = self.config.models[model_role]
        conversation = list(messages)
        usage = []
        parsed = None
        last_error = None
        attempts = self.config.pipeline.max_validation_retries + 1
        for attempt in range(1, attempts + 1):
            self._guard_budget()
            started = time.perf_counter()
            try:
                raw = self.client.chat(model.model, conversation, self.config.pipeline.temperature)
            except CloudError as error:
                last_error = str(error)
                self.events.write(
                    "llm_error",
                    run_id=run_id,
                    stage=stage,
                    model=model.model,
                    attempt=attempt,
                    error=str(error),
                )
                if error.status_code == 429 and attempt < attempts:
                    time.sleep(2)
                    continue
                break
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            safe_raw = self._scrub(raw)
            content, recovered = message_content(safe_raw.get("message", {}))
            cost = estimate_cost(safe_raw, model, self.config.usage_fields)
            if cost.known and cost.usd is not None:
                self.spent_usd += cost.usd
            record = {
                "run_id": run_id,
                "stage": stage,
                "model": model.model,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "recovered_from_thinking": recovered,
                "estimated_cost_usd": cost.usd,
                "input_tokens": cost.input_tokens,
                "cached_input_tokens": cost.cached_input_tokens,
                "output_tokens": cost.output_tokens,
                "cached_tokens_known": cost.cached_tokens_known,
                "cost_assumption": cost.assumption,
                "validation_error": None,
            }
            try:
                payload = extract_json(content)
                parsed = parser(payload)
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                record["validation_error"] = str(error)
                last_error = str(error)
                parsed = None
            usage.append(record)
            write_json(
                run_dir / f"{stage}-attempt-{attempt}.json",
                {
                    "record": record,
                    "request_messages": self._scrub(conversation),
                    "response": safe_raw,
                },
            )
            self.events.write("llm_call", **record)
            if parsed is not None:
                return parsed, usage, None
            if attempt < attempts:
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": content or "{}"},
                    {
                        "role": "user",
                        "content": (
                            "The response failed validation: "
                            f"{record['validation_error']}. Return only one corrected JSON object."
                        ),
                    },
                ]
        return None, usage, last_error

    def _parse_primary(self, payload: dict, tests: list[TestCase]) -> PrimaryResponse:
        parsed = PrimaryResponse.model_validate(payload)
        complete_primary(parsed, {test.id for test in tests})
        return parsed

    def _guard_budget(self) -> None:
        if self.spent_usd > self.config.cost_tracking.max_estimated_usd:
            raise CostCeilingExceeded(
                f"Estimated spend ${self.spent_usd:.4f} exceeded "
                f"${self.config.cost_tracking.max_estimated_usd:.2f}"
            )

    def _failed(self, run_id: str, tests: list[TestCase], usage: list[dict], reason: str) -> dict:
        result = {
            "run_id": run_id,
            "status": "failed",
            "degraded_reason": reason,
            "tests": [],
            "suite": None,
            "usage": usage,
            "estimated_cost_usd": round(sum(item["estimated_cost_usd"] or 0 for item in usage), 8),
            "input_case_ids": [test.id for test in tests],
        }
        write_json(self.output_dir / "runs" / run_id / "result.json", result)
        self.events.write("run_failed", run_id=run_id, reason=reason)
        return result

    def _input_record(self, tests: list[TestCase]) -> dict:
        messages = primary_messages(tests)
        return {
            "tests": [test.public_dict() for test in tests],
            "hidden_from_model": [
                {"id": test.id, "implementation": test.hidden_implementation}
                for test in tests
                if test.hidden_implementation
            ],
            "primary_prompt_sha256": _hash_messages(messages),
        }

    def _test_dict(self, item) -> dict:
        return {
            "test_case_id": item.test_case_id,
            "quality_score": item.quality_score,
            "spec_coverage_score": item.spec_coverage_score,
            "coverage_ratio": item.coverage_ratio,
            "correctness_score": item.correctness_score,
            "bug_detection_score": item.bug_detection_score,
            "llm_correctness_score": item.llm_correctness_score,
            "llm_bug_detection_score": item.llm_bug_detection_score,
            "assessments": [assessment.model_dump() for assessment in item.assessments],
            "adjustments": item.adjustments,
            "correctness_reasoning": item.correctness_reasoning,
            "bug_detection_reasoning": item.bug_detection_reasoning,
            "risks": item.risks,
        }

    def _scrub(self, value):
        if not self.api_key:
            return value
        encoded = json.dumps(value)
        return json.loads(encoded.replace(self.api_key, "[REDACTED]"))


def _hash_messages(messages: list[dict[str, str]]) -> str:
    raw = json.dumps(messages, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()
