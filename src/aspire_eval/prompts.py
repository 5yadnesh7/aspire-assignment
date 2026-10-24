import json

from aspire_eval.fixtures import SPEC, TestCase
from aspire_eval.rubric import REQUIREMENTS
from aspire_eval.schemas import PrimaryResponse


PRIMARY_SYSTEM = """You evaluate generated software tests against one payments specification.
Return one JSON object and no other text. Do not use Markdown.
Use only the supplied requirement IDs and test IDs.
Every test must contain every requirement exactly once.
Status must be one of: covered, partial, missing, incorrect, not_applicable.
Scores must be numbers from 0 to 10.
Evidence must be an exact short quote from that test. If you cannot quote it, do not use covered.
The artifact contains only a description, steps, and expected result. Do not penalize it for omitting executable code.
Use not_applicable when the test does not attempt the requirement.
Use incorrect when the test asserts behavior that conflicts with the specification.
Use partial when the test mentions the requirement but would not prove it.
"""


CRITIC_SYSTEM = """You audit a primary model's evaluation of generated software tests.
Return one JSON object and no other text. Do not use Markdown.
You may downgrade statuses and scores. You may not upgrade them.
correctness_adjustment and bug_detection_adjustment must be between -10 and 0.
Add an unsupported claim when covered evidence is not actually in the test.
Add a missed requirement when a relevant requirement was marked not_applicable.
Do not invent executable-code defects; the artifact has no executable code.
"""


def primary_messages(tests: list[TestCase]) -> list[dict[str, str]]:
    schema = {
        "evaluations": [
            {
                "test_case_id": "tc-id",
                "requirement_assessments": [
                    {
                        "requirement_id": "requirement-id",
                        "status": "covered|partial|missing|incorrect|not_applicable",
                        "evidence": "exact quote from the test",
                        "reasoning": "one sentence",
                    }
                ],
                "correctness_score": 0,
                "bug_detection_score": 0,
                "correctness_reasoning": "one sentence",
                "bug_detection_reasoning": "one sentence",
                "risks": ["specific risk"],
            }
        ]
    }
    requirements = [
        {"id": item.id, "statement": item.statement, "critical": item.critical}
        for item in REQUIREMENTS
    ]
    user = {
        "specification": SPEC,
        "requirements": requirements,
        "tests": [test.public_dict() for test in tests],
        "rules": [
            "Score correctness only for behavior the test attempts.",
            "A passing paraphrase of the specification is not automatically a strong bug detector.",
            "Timeout coverage requires the five-second boundary, not the word timeout alone.",
            "Approval coverage requires the above-SGD-10000 boundary, the second owner, and blocked settlement.",
            "Credit before debit requires that order, not merely a balance increase.",
        ],
        "response_schema": schema,
    }
    return [
        {"role": "system", "content": PRIMARY_SYSTEM},
        {"role": "user", "content": json.dumps(user, indent=2)},
    ]


def critic_messages(tests: list[TestCase], primary: PrimaryResponse) -> list[dict[str, str]]:
    schema = {
        "reviews": [
            {
                "test_case_id": "tc-id",
                "unsupported_claims": [{"requirement_id": "requirement-id", "issue": "why the evidence fails"}],
                "missed_requirements": [{"requirement_id": "requirement-id", "issue": "why it matters"}],
                "adjusted_statuses": [
                    {"requirement_id": "requirement-id", "status": "partial", "reason": "downgrade only"}
                ],
                "correctness_adjustment": 0,
                "bug_detection_adjustment": 0,
                "notes": "one sentence",
            }
        ]
    }
    user = {
        "specification": SPEC,
        "requirements": [
            {"id": item.id, "statement": item.statement} for item in REQUIREMENTS
        ],
        "tests": [test.public_dict() for test in tests],
        "primary_evaluation": primary.model_dump(),
        "response_schema": schema,
    }
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": json.dumps(user, indent=2)},
    ]
