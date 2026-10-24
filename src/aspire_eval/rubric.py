from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    id: str
    statement: str
    critical: bool = True


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        "fx-lock",
        "The FX rate is locked at the moment the top-up is initiated.",
    ),
    Requirement(
        "credit-before-debit",
        "The destination wallet is credited before the funding source is debited.",
    ),
    Requirement(
        "timeout-after-5s",
        "The payment-provider callback timeout occurs after 5 seconds.",
    ),
    Requirement(
        "rollback-on-timeout",
        "A timed-out provider callback rolls the transaction back.",
    ),
    Requirement(
        "neither-side-affected",
        "After rollback, neither the destination wallet nor the funding source is affected.",
    ),
    Requirement(
        "approval-above-10000",
        "Top-ups above SGD 10,000 require approval before settling.",
    ),
    Requirement(
        "second-owner-approval",
        "The required approval comes from a second business owner.",
    ),
    Requirement(
        "no-settle-before-approval",
        "A top-up that requires approval does not settle before that approval.",
    ),
)

REQUIREMENT_IDS: tuple[str, ...] = tuple(item.id for item in REQUIREMENTS)
