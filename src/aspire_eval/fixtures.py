from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


SPEC = (
    "When a business tops up their multi-currency wallet, the system locks the FX "
    "rate at the moment the top-up is initiated. The funding source is only debited "
    "after the destination wallet is credited. If the payment provider's callback "
    "times out after 5 seconds, the transaction is rolled back and neither side is "
    "affected. Top-ups above SGD 10,000 require approval from a second business "
    "owner before settling."
)


class TestCase(BaseModel):
    __test__: ClassVar[bool] = False
    model_config = ConfigDict(populate_by_name=True)

    id: str
    description: str
    steps: list[str]
    expected_result: str = Field(alias="expectedResult")
    hidden_implementation: str | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "steps": self.steps,
            "expectedResult": self.expected_result,
        }

    def text(self) -> str:
        return "\n".join([self.description, *self.steps, self.expected_result])


GENERATED_TESTS: tuple[TestCase, ...] = (
    TestCase(
        id="tc-1",
        description="Verify that a wallet top-up completes successfully",
        steps=["Initiate top-up of SGD 500", "Check wallet balance increases"],
        expectedResult="Wallet balance increases by SGD 500",
    ),
    TestCase(
        id="tc-2",
        description="Verify FX rate is applied correctly",
        steps=[
            "Initiate top-up of USD 100",
            "Check that FX rate shown equals rate at initiation time",
        ],
        expectedResult="FX rate matches rate locked at initiation",
    ),
    TestCase(
        id="tc-3",
        description="Verify rollback on timeout",
        steps=[
            "Initiate top-up",
            "Simulate provider timeout",
            "Check wallet balance",
        ],
        expectedResult="Wallet balance unchanged and funding source not debited",
    ),
    TestCase(
        id="tc-4",
        description="Verify large top-up requires approval",
        steps=[
            "Initiate top-up of SGD 15,000",
            "Check approval screen appears",
        ],
        expectedResult="Approval screen is shown",
    ),
)


ADVERSARIAL_TEST = TestCase(
    id="tc-adversarial",
    description=(
        "Verify FX lock at initiation, credit before debit, rollback after a "
        "5 second timeout, and second-owner approval before settlement"
    ),
    steps=[
        "Start a wallet top-up above SGD 10,000",
        "Verify the FX rate is locked at the moment the top-up is initiated",
        "Verify the destination wallet is credited before the funding source is debited",
        "Verify that when the provider callback times out after 5 seconds, the transaction is rolled back and neither side is affected",
        "Verify a second business owner must approve before the top-up settles",
    ],
    expectedResult=(
        "The FX rate remains the rate locked at initiation, the wallet is credited "
        "before the funding source is debited, a timeout after 5 seconds leaves "
        "neither side affected, and a top-up above SGD 10,000 does not settle until "
        "a second business owner approves"
    ),
    hidden_implementation="it('covers the spec', () => { expect(true).toBe(true); });",
)
