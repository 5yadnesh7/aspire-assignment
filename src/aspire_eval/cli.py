import argparse
import json
import os
import sys

from aspire_eval.client import CostCeilingExceeded, OllamaCloudClient
from aspire_eval.config import load_config
from aspire_eval.fixtures import ADVERSARIAL_TEST, GENERATED_TESTS, SPEC
from aspire_eval.observability import EventLog, write_json
from aspire_eval.pipeline import EvaluationPipeline
from aspire_eval.prompts import CRITIC_SYSTEM, PRIMARY_SYSTEM
from aspire_eval.reporting import format_run, format_summary, summarize
from aspire_eval.rubric import REQUIREMENTS
from aspire_eval.schemas import CriticResponse, PrimaryResponse


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate generated wallet top-up tests")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-adversarial", action="store_true")
    parser.add_argument(
        "--show",
        nargs="*",
        help="Print saved results without calling a model. Optionally name run-02 or adversarial-01.",
    )
    args = parser.parse_args()

    paths, config = load_config()
    if args.show is not None:
        _show_saved(paths.outputs, args.show, config.pipeline.quality_target)
        return
    api_key = os.environ.get(config.provider.api_key_env, "")
    client = OllamaCloudClient(config.provider.base_url, api_key)
    try:
        catalog = client.list_models()
        _check_catalog(config, catalog)
        if args.preflight_only:
            print("Ollama Cloud preflight succeeded")
            return
        full_submission = args.runs is None and not args.skip_adversarial
        output_dir = paths.outputs if full_submission else paths.outputs / "preview"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_static_artifacts(output_dir)
        events = EventLog(output_dir / "events.jsonl")
        pipeline = EvaluationPipeline(config, client, events, output_dir, api_key)
        run_count = args.runs or config.pipeline.evaluation_runs
        results = []
        for index in range(1, run_count + 1):
            run_id = f"run-{index:02d}"
            print(f"Starting {run_id}", flush=True)
            result = pipeline.evaluate(run_id, list(GENERATED_TESTS))
            results.append(result)
            print(format_run(result), flush=True)
        if not args.skip_adversarial:
            print("Starting adversarial-01", flush=True)
            adversarial = pipeline.evaluate("adversarial-01", [ADVERSARIAL_TEST])
            write_json(output_dir / "adversarial-result.json", adversarial)
            print(format_run(adversarial), flush=True)
        summary = summarize(results, config.pipeline.quality_target, events.events)
        summary["models"] = {
            role: {"model": model.model, "catalog_digest": catalog.get(model.model, model.catalog_digest)}
            for role, model in config.models.items()
        }
        summary["provider"] = {
            "name": config.provider.name,
            "base_url": config.provider.base_url,
            "pricing_source": config.provider.pricing_source,
            "pricing_retrieved_at": config.provider.pricing_retrieved_at,
            "native_schema_constrained_output": config.provider.native_schema_constrained_output,
        }
        write_json(output_dir / "summary.json", summary)
        print(format_summary(summary), flush=True)
        if not full_submission:
            print(f"Preview saved separately at {output_dir}", flush=True)
    except CostCeilingExceeded as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        client.close()


def _show_saved(output_dir, requested: list[str], quality_target: float) -> None:
    run_dir = output_dir / "runs"
    available = sorted(path for path in run_dir.glob("*/result.json")) if run_dir.exists() else []
    selected = available
    if requested:
        selected = []
        for name in requested:
            path = run_dir / name / "result.json"
            if not path.exists():
                raise SystemExit(f"No saved result for {name}")
            selected.append(path)
    if not selected:
        raise SystemExit("No saved runs found in outputs/runs")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in selected]
    for result in results:
        print(format_run(result), flush=True)
    if requested:
        return
    comparable = [item for item in results if not str(item.get("run_id", "")).startswith("adversarial")]
    events = [
        {"event": "llm_call", **usage}
        for item in comparable
        for usage in (item.get("usage") or [])
    ]
    print(format_summary(summarize(comparable, quality_target, events)), flush=True)


def _check_catalog(config, catalog: dict[str, str]) -> None:
    missing = [model.model for model in config.models.values() if model.model not in catalog]
    if missing:
        raise SystemExit(f"Configured cloud models are unavailable: {', '.join(missing)}")
    for model in config.models.values():
        print(f"Cloud model available: {model.model} digest={catalog[model.model]}")


def _write_static_artifacts(output_dir) -> None:
    prompt_dir = output_dir / "prompts"
    schema_dir = output_dir / "schemas"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "primary_system.txt").write_text(PRIMARY_SYSTEM, encoding="utf-8")
    (prompt_dir / "critic_system.txt").write_text(CRITIC_SYSTEM, encoding="utf-8")
    (output_dir / "specification.txt").write_text(SPEC, encoding="utf-8")
    write_json(schema_dir / "primary.json", PrimaryResponse.model_json_schema())
    write_json(schema_dir / "critic.json", CriticResponse.model_json_schema())
    write_json(
        output_dir / "rubric.json",
        [{"id": item.id, "statement": item.statement, "critical": item.critical} for item in REQUIREMENTS],
    )
