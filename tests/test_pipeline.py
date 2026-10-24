import json
from pathlib import Path

from aspire_eval.client import CloudError
from aspire_eval.config import load_config
from aspire_eval.fixtures import GENERATED_TESTS
from aspire_eval.observability import EventLog
from aspire_eval.pipeline import EvaluationPipeline
from aspire_eval.rubric import REQUIREMENT_IDS

ROOT = Path(__file__).parents[1]


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(self, model: str, messages: list[dict], temperature: float) -> dict:
        del model, messages, temperature
        return self.responses.pop(0)


def _response(payload: dict, secret: str = "") -> dict:
    return {
        "message": {"role": "assistant", "content": json.dumps(payload) + secret},
        "prompt_eval_count": 100,
        "eval_count": 40,
    }


def _primary(test_id: str) -> dict:
    return {
        "evaluations": [
            {
                "test_case_id": test_id,
                "requirement_assessments": [
                    {
                        "requirement_id": requirement_id,
                        "status": "not_applicable",
                        "evidence": "",
                        "reasoning": "not attempted",
                    }
                    for requirement_id in REQUIREMENT_IDS
                ],
                "correctness_score": 6,
                "bug_detection_score": 6,
                "correctness_reasoning": "narrow",
                "bug_detection_reasoning": "narrow",
                "risks": [],
            }
        ]
    }


def test_invalid_json_is_retried_and_secret_is_scrubbed(tmp_path: Path) -> None:
    _, config = load_config(ROOT)
    secret = "super-secret-value"
    client = FakeClient(
        [
            _response({"not": "valid"}, secret),
            _response(_primary("tc-1")),
            _response({"reviews": []}),
        ]
    )
    events = EventLog(tmp_path / "events.jsonl")
    pipeline = EvaluationPipeline(config, client, events, tmp_path, secret)
    result = pipeline.evaluate("run-01", [GENERATED_TESTS[0]])

    assert result["status"] == "completed"
    saved = json.loads((tmp_path / "runs" / "run-01" / "primary_judge-attempt-1.json").read_text(encoding="utf-8"))
    assert secret not in json.dumps(saved)
    assert saved["record"]["validation_error"]
    assert result["tests"][0]["quality_score"] == 12


def test_provider_error_is_kept_on_the_run(tmp_path: Path) -> None:
    _, config = load_config(ROOT)

    class FailingClient:
        def chat(self, model: str, messages: list[dict], temperature: float) -> dict:
            del model, messages, temperature
            raise CloudError(402, "this model is not included in your free usage")

    events = EventLog(tmp_path / "events.jsonl")
    pipeline = EvaluationPipeline(config, FailingClient(), events, tmp_path, "")
    result = pipeline.evaluate("run-01", [GENERATED_TESTS[0]])
    assert result["status"] == "failed"
    assert "HTTP 402" in result["degraded_reason"]
