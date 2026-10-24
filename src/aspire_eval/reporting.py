import math
import statistics
from collections import Counter


def summarize(results: list[dict], quality_target: float, events: list[dict]) -> dict:
    completed = [item for item in results if item.get("suite")]
    gate_scores = [item["suite"]["gate_score"] for item in completed]
    calibration = calibrate(gate_scores, quality_target)
    threshold = calibration["recommended_threshold"]
    verdicts = []
    for item in completed:
        blocked = item["suite"]["gate_score"] < threshold or bool(item["suite"]["absolute_misses"])
        verdicts.append({"run_id": item["run_id"], "verdict": "BLOCK" if blocked else "PASS"})
    calls = [item for item in events if item.get("event") == "llm_call"]
    validation_failures = sum(1 for item in calls if item.get("validation_error"))
    known_costs = [item["estimated_cost_usd"] for item in calls if item.get("estimated_cost_usd") is not None]
    return {
        "completed_runs": len(completed),
        "failed_runs": len(results) - len(completed),
        "gate_scores": gate_scores,
        "per_test_quality": _per_test(completed),
        "requirement_agreement": _agreement(completed),
        "calibration": calibration,
        "verdicts_at_recommended_threshold": verdicts,
        "verdict_flip": len({item["verdict"] for item in verdicts}) > 1,
        "llm_calls": len(calls),
        "validation_failure_count": validation_failures,
        "json_validity_rate": round(1 - validation_failures / len(calls), 4) if calls else None,
        "estimated_cost_usd": round(sum(known_costs), 8),
        "cost_by_model": _cost_by_model(calls),
        "latency_ms": _latency(calls),
    }


def calibrate(scores: list[float], quality_target: float) -> dict:
    if not scores:
        return {
            "quality_target": quality_target,
            "recommended_threshold": quality_target,
            "reason": "No completed runs were available to measure variance.",
            "absolute_rule": _absolute_rule(),
        }
    mean = statistics.fmean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    low = min(scores)
    high = max(scores)
    if low <= quality_target <= high:
        recommended = max(0, math.floor(low - stdev))
        reason = (
            "The quality target falls inside the observed score range. "
            "The threshold is moved below that noise band so the same artifact does not flap."
        )
    else:
        recommended = quality_target
        reason = "The quality target is outside the observed run-to-run score range."
    return {
        "quality_target": quality_target,
        "observed_mean": round(mean, 2),
        "observed_sample_stdev": round(stdev, 2),
        "observed_min": round(low, 2),
        "observed_max": round(high, 2),
        "observed_range": round(high - low, 2),
        "recommended_threshold": recommended,
        "reason": reason,
        "absolute_rule": _absolute_rule(),
    }


def _per_test(results: list[dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for result in results:
        for test in result["tests"]:
            grouped.setdefault(test["test_case_id"], []).append(test["quality_score"])
    summary = {}
    for test_id, scores in grouped.items():
        summary[test_id] = _distribution(scores)
    return summary


def _agreement(results: list[dict]) -> dict:
    grouped: dict[str, list[str]] = {}
    for result in results:
        for requirement_id, status in result["suite"]["best_statuses"].items():
            grouped.setdefault(requirement_id, []).append(status)
    agreement = {}
    for requirement_id, statuses in grouped.items():
        counts = Counter(statuses)
        mode, count = counts.most_common(1)[0]
        agreement[requirement_id] = {
            "mode": mode,
            "agreement_rate": round(count / len(statuses), 4),
            "statuses": statuses,
        }
    return agreement


def _distribution(scores: list[float]) -> dict:
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {
        "values": scores,
        "mean": round(statistics.fmean(scores), 2),
        "sample_stdev": round(stdev, 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "range": round(max(scores) - min(scores), 2),
    }


def _cost_by_model(calls: list[dict]) -> dict:
    totals: dict[str, float] = {}
    for call in calls:
        if call.get("estimated_cost_usd") is None:
            continue
        totals[call["model"]] = totals.get(call["model"], 0) + call["estimated_cost_usd"]
    return {model: round(total, 8) for model, total in totals.items()}


def _latency(calls: list[dict]) -> dict:
    values = sorted(float(call["latency_ms"]) for call in calls if call.get("latency_ms") is not None)
    if not values:
        return {}
    midpoint = values[len(values) // 2]
    return {"min": values[0], "median": midpoint, "max": values[-1]}


def format_run(result: dict) -> str:
    lines = [_line(), f"{result.get('run_id', 'run')}  status: {result.get('status', 'unknown')}"]
    if result.get("degraded_reason"):
        lines.append(f"Note: {result['degraded_reason']}")
    suite = result.get("suite") or {}
    if suite:
        misses = suite.get("absolute_misses") or []
        lines.append(
            f"Suite gate: {suite.get('gate_score')}    "
            f"Mean test quality: {suite.get('mean_quality_score')}"
        )
        lines.append("Critical misses: " + (", ".join(misses) if misses else "none"))
    lines.append(f"Run cost USD: {result.get('estimated_cost_usd')}")
    for test in result.get("tests") or []:
        lines.extend(["", _line("-"), _format_test(test)])
    lines.append(_line())
    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    calibration = summary.get("calibration") or {}
    lines = [
        _line(),
        "VARIANCE AND CI THRESHOLD",
        f"Gate scores: {summary.get('gate_scores')}",
        f"Mean: {calibration.get('observed_mean')}    "
        f"Sample standard deviation: {calibration.get('observed_sample_stdev')}    "
        f"Range: {calibration.get('observed_range')}",
        f"Recommended threshold: {calibration.get('recommended_threshold')}",
        calibration.get("reason", ""),
        calibration.get("absolute_rule", ""),
        f"Verdict flip across these runs: {summary.get('verdict_flip')}",
    ]
    for verdict in summary.get("verdicts_at_recommended_threshold") or []:
        lines.append(f"  {verdict['run_id']}: {verdict['verdict']}")
    lines.extend(
        [
            f"JSON validity: {summary.get('json_validity_rate')} "
            f"across {summary.get('llm_calls')} model calls",
            f"Estimated cost USD: {summary.get('estimated_cost_usd')}",
            f"Cost by model: {summary.get('cost_by_model')}",
            _line(),
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def _format_test(test: dict) -> str:
    lines = [
        (
            f"{test['test_case_id']}    quality {test['quality_score']}    "
            f"correctness {test['correctness_score']}    "
            f"bug detection {test['bug_detection_score']}    "
            f"spec coverage {test['spec_coverage_score']}"
        )
    ]
    for assessment in test.get("assessments") or []:
        model_status = assessment.get("llm_status") or ""
        changed = ""
        if model_status and model_status != assessment.get("status"):
            changed = f"    model said {model_status}"
        lines.append(f"  {assessment['requirement_id']:<28} {assessment.get('status')}{changed}")
        if assessment.get("reasoning"):
            lines.append(f"    {assessment['reasoning']}")
    if test.get("adjustments"):
        lines.append("  Adjustments:")
        lines.extend(f"    - {note}" for note in test["adjustments"])
    if test.get("correctness_reasoning"):
        lines.append(f"  Correctness: {test['correctness_reasoning']}")
    if test.get("bug_detection_reasoning"):
        lines.append(f"  Bug detection: {test['bug_detection_reasoning']}")
    return "\n".join(lines)


def _line(character: str = "=") -> str:
    return character * 72


def _absolute_rule() -> str:
    return (
        "Block when any critical requirement has a best status of missing or incorrect, "
        "regardless of the numeric threshold."
    )
