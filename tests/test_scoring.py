from pathlib import Path

from aspire_eval.config import load_config
from aspire_eval.fixtures import ADVERSARIAL_TEST, GENERATED_TESTS, TestCase
from aspire_eval.prompts import primary_messages
from aspire_eval.reporting import calibrate, format_run
from aspire_eval.rubric import REQUIREMENT_IDS
from aspire_eval.schemas import PrimaryResponse, TestReview
from aspire_eval.scoring import best_requirement_status, maximum_status, reconcile, suite_gate_score

ROOT = Path(__file__).parents[1]


def _evaluation(test_id: str, statuses: dict[str, str], evidence: str = "", correctness: float = 8, bug: float = 8):
    assessments = []
    for requirement_id in REQUIREMENT_IDS:
        assessments.append(
            {
                "requirement_id": requirement_id,
                "status": statuses.get(requirement_id, "not_applicable"),
                "evidence": evidence,
                "reasoning": "checked",
            }
        )
    return PrimaryResponse.model_validate(
        {
            "evaluations": [
                {
                    "test_case_id": test_id,
                    "requirement_assessments": assessments,
                    "correctness_score": correctness,
                    "bug_detection_score": bug,
                    "correctness_reasoning": "checked",
                    "bug_detection_reasoning": "checked",
                    "risks": [],
                }
            ]
        }
    ).evaluations[0]


def test_adversarial_wording_passes_every_deterministic_cap() -> None:
    assert all(maximum_status(requirement_id, ADVERSARIAL_TEST.text()) is None for requirement_id in REQUIREMENT_IDS)


def test_hidden_implementation_is_not_sent_to_the_model() -> None:
    content = primary_messages([ADVERSARIAL_TEST])[1]["content"]
    assert "expect(true)" not in content
    assert ADVERSARIAL_TEST.hidden_implementation not in content


def test_exact_boundary_approval_is_incorrect() -> None:
    case = TestCase(
        id="boundary",
        description="Approval at the threshold",
        steps=["Initiate top-up of SGD 10,000"],
        expectedResult="Approval is required",
    )
    assert maximum_status("approval-above-10000", case.text()) == "incorrect"


def test_official_cases_cannot_claim_unsupported_coverage() -> None:
    assert maximum_status("timeout-after-5s", GENERATED_TESTS[2].text()) == "partial"
    assert maximum_status("approval-above-10000", GENERATED_TESTS[3].text()) == "partial"
    assert maximum_status("second-owner-approval", GENERATED_TESTS[3].text()) == "missing"
    assert maximum_status("no-settle-before-approval", GENERATED_TESTS[3].text()) == "missing"
    assert maximum_status("fx-lock", GENERATED_TESTS[1].text()) is None


def test_ungrounded_covered_claim_is_downgraded() -> None:
    _, config = load_config(ROOT)
    evaluation = _evaluation(
        "tc-2",
        {"fx-lock": "covered"},
        evidence="this evidence was not written in the test",
    )
    result = reconcile(GENERATED_TESTS[1], evaluation, None, config.pipeline.score_weights)
    status = next(item for item in result.assessments if item.requirement_id == "fx-lock")
    assert status.status == "partial"
    assert status.llm_status == "covered"


def test_critic_cannot_upgrade_a_status() -> None:
    _, config = load_config(ROOT)
    evaluation = _evaluation("tc-2", {"fx-lock": "partial"}, evidence="FX rate matches rate locked at initiation")
    review = TestReview(
        test_case_id="tc-2",
        adjusted_statuses=[{"requirement_id": "fx-lock", "status": "covered", "reason": "too generous"}],
    )
    result = reconcile(GENERATED_TESTS[1], evaluation, review, config.pipeline.score_weights)
    status = next(item.status for item in result.assessments if item.requirement_id == "fx-lock")
    assert status == "partial"
    assert any("Ignored critic upgrade" in note for note in result.adjustments)


def test_grounded_spec_paraphrase_can_score_too_high() -> None:
    _, config = load_config(ROOT)
    evidence = "The FX rate remains the rate locked at initiation"
    evaluation = _evaluation(
        ADVERSARIAL_TEST.id,
        {requirement_id: "covered" for requirement_id in REQUIREMENT_IDS},
        evidence=evidence,
        correctness=10,
        bug=10,
    )
    result = reconcile(ADVERSARIAL_TEST, evaluation, None, config.pipeline.score_weights)
    assert result.quality_score == 100
    assert suite_gate_score(best_requirement_status([result])) == 100


def test_threshold_moves_below_observed_noise() -> None:
    calibration = calibrate([70, 80], 75)
    assert calibration["recommended_threshold"] == 62


def test_terminal_report_includes_scores_and_reasons() -> None:
    report = format_run(
        {
            "run_id": "run-01",
            "status": "completed",
            "estimated_cost_usd": 0.01,
            "suite": {
                "gate_score": 50,
                "mean_quality_score": 40,
                "absolute_misses": ["credit-before-debit"],
            },
            "tests": [
                {
                    "test_case_id": "tc-1",
                    "quality_score": 12,
                    "correctness_score": 2,
                    "bug_detection_score": 2,
                    "spec_coverage_score": 0,
                    "assessments": [
                        {
                            "requirement_id": "fx-lock",
                            "status": "not_applicable",
                            "llm_status": "covered",
                            "reasoning": "No FX behavior is asserted.",
                        }
                    ],
                    "adjustments": ["Deterministic cap changed fx-lock from covered to not_applicable"],
                    "correctness_reasoning": "The balance check is narrow.",
                    "bug_detection_reasoning": "It would miss an ordering bug.",
                }
            ],
        }
    )
    assert "Suite gate: 50" in report
    assert "quality 12" in report
    assert "model said covered" in report
    assert "No FX behavior is asserted." in report
    assert "ordering bug" in report


def test_stable_scores_keep_the_quality_target() -> None:
    assert calibrate([40, 50], 75)["recommended_threshold"] == 75
    assert calibrate([80, 90], 75)["recommended_threshold"] == 75
