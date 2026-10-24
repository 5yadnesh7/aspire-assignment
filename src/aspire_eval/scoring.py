import re
from dataclasses import dataclass, field

from aspire_eval.config import ScoreWeights
from aspire_eval.fixtures import TestCase
from aspire_eval.rubric import REQUIREMENTS, Requirement
from aspire_eval.schemas import (
    RequirementAssessment,
    TestEvaluation,
    TestReview,
)

STATUS_WEIGHT = {
    "covered": 1.0,
    "partial": 0.5,
    "missing": 0.0,
    "incorrect": 0.0,
    "not_applicable": 0.0,
}
STATUS_RANK = {"incorrect": 0, "missing": 1, "partial": 2, "covered": 3}
POSITIVE_RANK = {"covered": 3, "partial": 2, "missing": 1, "incorrect": 0}


@dataclass
class ReconciledTest:
    test_case_id: str
    assessments: list[RequirementAssessment]
    quality_score: float
    spec_coverage_score: float
    coverage_ratio: float
    correctness_score: float
    bug_detection_score: float
    llm_correctness_score: float
    llm_bug_detection_score: float
    adjustments: list[str] = field(default_factory=list)
    correctness_reasoning: str = ""
    bug_detection_reasoning: str = ""
    risks: list[str] = field(default_factory=list)


def maximum_status(requirement_id: str, text: str) -> str | None:
    checkers = {
        "fx-lock": _cap_fx,
        "credit-before-debit": _cap_credit,
        "timeout-after-5s": _cap_timeout,
        "rollback-on-timeout": _cap_rollback,
        "neither-side-affected": _cap_neither,
        "approval-above-10000": _cap_approval,
        "second-owner-approval": _cap_second_owner,
        "no-settle-before-approval": _cap_settlement,
    }
    return checkers[requirement_id](text)


def reconcile(
    test: TestCase,
    evaluation: TestEvaluation,
    review: TestReview | None,
    weights: ScoreWeights,
) -> ReconciledTest:
    by_id: dict[str, RequirementAssessment] = {}
    for item in evaluation.requirement_assessments:
        copy = item.model_copy(deep=True)
        copy.llm_status = item.status
        by_id[item.requirement_id] = copy
    notes: list[str] = []
    if review is not None:
        _apply_review(by_id, review, notes)
    _apply_caps(test, by_id, notes)
    _apply_grounding(test, by_id, notes)

    correctness = evaluation.correctness_score
    bug_detection = evaluation.bug_detection_score
    if review is not None:
        correctness = _clamp(correctness + review.correctness_adjustment)
        bug_detection = _clamp(bug_detection + review.bug_detection_adjustment)
        if review.correctness_adjustment:
            notes.append(f"Critic correctness adjustment {review.correctness_adjustment}")
        if review.bug_detection_adjustment:
            notes.append(f"Critic bug-detection adjustment {review.bug_detection_adjustment}")

    applicable = [
        item for item in by_id.values() if item.status != "not_applicable"
    ]
    if not applicable:
        correctness = min(correctness, 2)
        bug_detection = min(bug_detection, 2)
        notes.append("Scores capped because the test attempts none of the atomic requirements")
    if any(item.status == "incorrect" for item in applicable):
        correctness = min(correctness, 4)
        notes.append("Correctness capped at 4 because an attempted requirement is incorrect")
    if applicable and not any(item.status == "covered" for item in applicable):
        bug_detection = min(bug_detection, 5)
        notes.append("Bug detection capped at 5 because no attempted requirement is fully covered")

    coverage_ratio = (
        sum(STATUS_WEIGHT[item.status] for item in applicable) / len(applicable)
        if applicable
        else 0
    )
    spec_ratio = sum(STATUS_WEIGHT[by_id[item.id].status] for item in REQUIREMENTS) / len(REQUIREMENTS)
    quality = 100 * (
        weights.coverage * coverage_ratio
        + weights.correctness * correctness / 10
        + weights.bug_detection * bug_detection / 10
    )
    return ReconciledTest(
        test_case_id=test.id,
        assessments=[by_id[item.id] for item in REQUIREMENTS],
        quality_score=round(quality, 2),
        spec_coverage_score=round(100 * spec_ratio, 2),
        coverage_ratio=round(coverage_ratio, 4),
        correctness_score=round(correctness, 2),
        bug_detection_score=round(bug_detection, 2),
        llm_correctness_score=evaluation.correctness_score,
        llm_bug_detection_score=evaluation.bug_detection_score,
        adjustments=notes,
        correctness_reasoning=evaluation.correctness_reasoning,
        bug_detection_reasoning=evaluation.bug_detection_reasoning,
        risks=evaluation.risks,
    )


def best_requirement_status(tests: list[ReconciledTest]) -> dict[str, str]:
    best: dict[str, str] = {}
    for requirement in REQUIREMENTS:
        statuses = [
            next(item.status for item in test.assessments if item.requirement_id == requirement.id)
            for test in tests
        ]
        considered = [status for status in statuses if status != "not_applicable"]
        best[requirement.id] = _best_status(considered or ["missing"])
    return best


def suite_gate_score(best_statuses: dict[str, str]) -> float:
    total = sum(STATUS_WEIGHT[status] for status in best_statuses.values())
    return round(100 * total / len(best_statuses), 2)


def absolute_misses(best_statuses: dict[str, str]) -> list[str]:
    return [
        requirement.id
        for requirement in REQUIREMENTS
        if requirement.critical and best_statuses[requirement.id] in {"missing", "incorrect"}
    ]


def _apply_review(
    by_id: dict[str, RequirementAssessment],
    review: TestReview,
    notes: list[str],
) -> None:
    for claim in review.unsupported_claims:
        current = by_id.get(claim.requirement_id)
        if current is None:
            continue
        downgraded = {"covered": "partial", "partial": "missing"}.get(current.status)
        if downgraded:
            current.status = downgraded
            notes.append(f"Unsupported {claim.requirement_id}: {claim.issue}")
    for missed in review.missed_requirements:
        current = by_id.get(missed.requirement_id)
        if current and current.status == "not_applicable":
            current.status = "missing"
            notes.append(f"Missed {missed.requirement_id}: {missed.issue}")
    for adjustment in review.adjusted_statuses:
        current = by_id.get(adjustment.requirement_id)
        if current is None or adjustment.status == "not_applicable":
            continue
        if _is_downgrade(current.status, adjustment.status):
            current.status = adjustment.status
            notes.append(
                f"Critic downgraded {adjustment.requirement_id} to {adjustment.status}: {adjustment.reason}"
            )
        elif adjustment.status != current.status:
            notes.append(f"Ignored critic upgrade for {adjustment.requirement_id} to {adjustment.status}")


def _apply_caps(test: TestCase, by_id: dict[str, RequirementAssessment], notes: list[str]) -> None:
    for requirement_id, assessment in by_id.items():
        cap = maximum_status(requirement_id, test.text())
        if cap is None:
            continue
        updated = _tighten(assessment.status, cap)
        if updated != assessment.status:
            notes.append(f"Deterministic cap changed {requirement_id} from {assessment.status} to {updated}")
            assessment.status = updated


def _apply_grounding(
    test: TestCase,
    by_id: dict[str, RequirementAssessment],
    notes: list[str],
) -> None:
    normalized_text = " ".join(test.text().split()).casefold()
    for assessment in by_id.values():
        if assessment.status != "covered":
            continue
        evidence = " ".join(assessment.evidence.split()).casefold()
        if len(evidence) < 12 or evidence not in normalized_text:
            assessment.status = "partial"
            notes.append(f"Ungrounded covered evidence for {assessment.requirement_id}")


def _best_status(statuses: list[str]) -> str:
    return max(statuses, key=lambda status: POSITIVE_RANK[status])


def _is_downgrade(current: str, proposed: str) -> bool:
    if current == "not_applicable":
        return proposed in {"partial", "missing", "incorrect"}
    if proposed == "incorrect" and current != "incorrect":
        return True
    return STATUS_RANK.get(proposed, -1) < STATUS_RANK.get(current, -1)


def _tighten(current: str, cap: str) -> str:
    if current == "not_applicable":
        return current
    if cap == "incorrect":
        return "incorrect"
    if STATUS_RANK[current] > STATUS_RANK[cap]:
        return cap
    return current


def _clamp(value: float) -> float:
    return min(10, max(0, value))


def _cap_fx(text: str) -> str | None:
    has_fx = re.search(r"\b(fx|foreign exchange|exchange rate)\b", text, re.IGNORECASE)
    has_initiation = re.search(r"\binitiat", text, re.IGNORECASE)
    has_lock = re.search(r"\block", text, re.IGNORECASE) or "rate at initiation" in text.casefold()
    if has_fx and has_initiation and has_lock:
        return None
    if has_fx:
        return "partial"
    return "missing"


def _cap_credit(text: str) -> str | None:
    credit = re.search(r"\bcredit", text, re.IGNORECASE)
    debit = re.search(r"\bdebit", text, re.IGNORECASE)
    if credit and debit and credit.start() < debit.start():
        return None
    if re.search(r"\b(credit|balance)\b", text, re.IGNORECASE):
        return "partial"
    return "missing"


def _cap_timeout(text: str) -> str | None:
    durations = []
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(milliseconds|millisecond|seconds|second|secs|sec|ms)\b",
        text,
        re.IGNORECASE,
    ):
        value = float(match.group(1))
        unit = match.group(2).lower()
        durations.append(value / 1000 if unit.startswith("milli") or unit == "ms" else value)
    if any(abs(duration - 5) < 1e-6 for duration in durations):
        return None
    if durations:
        return "incorrect"
    if "timeout" in text.casefold():
        return "partial"
    return "missing"


def _cap_rollback(text: str) -> str | None:
    has_rollback = re.search(r"\brollback\b|\brolls back\b|\brolled back\b", text, re.IGNORECASE)
    has_timeout = "timeout" in text.casefold()
    if has_rollback and has_timeout:
        return None
    if has_rollback or has_timeout:
        return "partial"
    return "missing"


def _cap_neither(text: str) -> str | None:
    has_wallet = re.search(r"\b(wallet|balance)\b", text, re.IGNORECASE)
    has_funding = re.search(r"\bfunding source\b|\bdebit", text, re.IGNORECASE)
    has_effect = re.search(r"\bneither side\b|\bunchanged\b|\bnot debited\b|\bnot affected\b", text, re.IGNORECASE)
    if has_wallet and has_funding and has_effect:
        return None
    if has_wallet or has_funding:
        return "partial"
    return "missing"


def _cap_approval(text: str) -> str | None:
    if re.search(r"\b(above|over|greater than|more than)\s+SGD\s*10,?000\b", text, re.IGNORECASE):
        return None
    amounts = [
        float(match.group(1).replace(",", ""))
        for match in re.finditer(r"\bSGD\s*([0-9][0-9,]*(?:\.\d+)?)", text, re.IGNORECASE)
    ]
    expects_approval = "approv" in text.casefold()
    if expects_approval and any(abs(amount - 10000) < 1e-6 for amount in amounts):
        return "incorrect"
    if expects_approval and any(amount > 10000 for amount in amounts):
        return "partial"
    if expects_approval:
        return "partial"
    return "missing"


def _cap_second_owner(text: str) -> str | None:
    if re.search(r"\bsecond\s+(?:business\s+)?owner\b", text, re.IGNORECASE):
        return None
    return "missing"


def _cap_settlement(text: str) -> str | None:
    has_approval = "approv" in text.casefold()
    has_gate = re.search(r"\bbefore\b[\s\S]{0,80}\bsettl", text, re.IGNORECASE) or re.search(
        r"\b(?:does not|doesn't|not)\s+settl",
        text,
        re.IGNORECASE,
    )
    if has_approval and has_gate:
        return None
    if "settl" in text.casefold():
        return "partial"
    return "missing"


def requirement_by_id(requirement_id: str) -> Requirement:
    return next(item for item in REQUIREMENTS if item.id == requirement_id)
