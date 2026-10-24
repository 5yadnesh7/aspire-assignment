import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aspire_eval.rubric import REQUIREMENT_IDS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RequirementAssessment(StrictModel):
    requirement_id: str
    status: str
    evidence: str = ""
    reasoning: str = ""

    llm_status: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def known_status(cls, value: str) -> str:
        return _normalize_status(value)

    @field_validator("requirement_id")
    @classmethod
    def known_requirement(cls, value: str) -> str:
        if value not in REQUIREMENT_IDS:
            raise ValueError(f"Unknown requirement id: {value}")
        return value


class TestEvaluation(StrictModel):
    __test__: ClassVar[bool] = False
    test_case_id: str
    requirement_assessments: list[RequirementAssessment]
    correctness_score: float = Field(ge=0, le=10)
    bug_detection_score: float = Field(ge=0, le=10)
    correctness_reasoning: str = ""
    bug_detection_reasoning: str = ""
    risks: list[str] = Field(default_factory=list)


class PrimaryResponse(StrictModel):
    evaluations: list[TestEvaluation]


class StatusAdjustment(StrictModel):
    requirement_id: str
    status: str
    reason: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def known_status(cls, value: str) -> str:
        return _normalize_status(value)


class ReviewNote(StrictModel):
    requirement_id: str
    issue: str


class TestReview(StrictModel):
    __test__: ClassVar[bool] = False
    test_case_id: str
    unsupported_claims: list[ReviewNote] = Field(default_factory=list)
    missed_requirements: list[ReviewNote] = Field(default_factory=list)
    adjusted_statuses: list[StatusAdjustment] = Field(default_factory=list)
    correctness_adjustment: float = Field(default=0, ge=-10, le=0)
    bug_detection_adjustment: float = Field(default=0, ge=-10, le=0)
    notes: str = ""


class CriticResponse(StrictModel):
    reviews: list[TestReview]


def _normalize_status(value: str) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    allowed = {"covered", "partial", "missing", "incorrect", "not_applicable"}
    if normalized not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")
    return normalized


def complete_primary(response: PrimaryResponse, test_ids: set[str]) -> None:
    seen = [item.test_case_id for item in response.evaluations]
    if set(seen) != test_ids or len(seen) != len(test_ids):
        raise ValueError(f"Primary response test ids {seen} did not match {sorted(test_ids)}")
    for evaluation in response.evaluations:
        ids = [item.requirement_id for item in evaluation.requirement_assessments]
        if sorted(ids) != sorted(REQUIREMENT_IDS):
            raise ValueError(
                f"{evaluation.test_case_id} assessed {ids}; expected every requirement exactly once"
            )


def strip_model_wrappers(text: str) -> str:
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
