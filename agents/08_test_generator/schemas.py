"""Pydantic schemas for the test-generator agent.

Two models:
- `TestExecutionResult`: the structured feedback the sandbox subprocess
  gives back to the model after a test run. Making this a Pydantic
  model (not a plain dict) means the ReAct loop's `execute_test_code`
  tool has a stable, documented shape the model can reason about.
- `GeneratedTest`: the final `output_type` returned from the Agent's
  Runner. Carries the finished test code, count of tests, whether
  they all passed, and the last execution result for evidence.

A model validator on `GeneratedTest` cross-checks that `all_passing`
matches the underlying `final_result.tests_failed == 0 and not
timed_out`. Catches an agent that claims success while its last real
execution said otherwise; regression guard against a future prompt
change that lets the model set `all_passing=True` on wishful thinking.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TestExecutionResult(BaseModel):
    """One run of the generated tests inside the sandbox subprocess.

    `exit_code` is pytest's exit code (0 = all passed, 1 = failures,
    2 = interrupted / collection errors, 5 = no tests collected).
    `timed_out` is set when the subprocess hit the wall-clock timeout
    (in which case exit_code is -1 and stdout/stderr may be partial).
    """

    # The name starts with "Test" only because it describes a test
    # run; this tells pytest not to try to collect it as a test class.
    __test__ = False

    exit_code: int = Field(..., description="pytest's return code; -1 on timeout.")
    stdout: str = Field("", description="Captured pytest stdout, verbatim.")
    stderr: str = Field("", description="Captured pytest stderr, verbatim.")
    timed_out: bool = Field(
        False, description="True if the subprocess exceeded its wall-clock timeout."
    )
    tests_passed: int = Field(
        0, ge=0, description="Count of PASSED tests parsed from pytest's summary line."
    )
    tests_failed: int = Field(
        0, ge=0, description="Count of FAILED tests parsed from pytest's summary line."
    )


class GeneratedTest(BaseModel):
    """The finished test suite the agent produced for one source file."""

    target_module: str = Field(
        ..., description="Path or module name the generated tests target."
    )
    test_code: str = Field(
        ..., description="The final generated test file content, ready to write to disk."
    )
    tests_added: int = Field(
        ..., ge=0, description="Count of `def test_*` functions in the generated code."
    )
    iterations_used: int = Field(
        ...,
        ge=1,
        description=(
            "How many generate-execute cycles the agent used. Bounded by "
            "max_iterations passed to generate_tests()."
        ),
    )
    all_passing: bool = Field(
        ...,
        description=(
            "True IFF the final sandbox run had zero failures and did not "
            "time out. Cross-checked by a model validator against "
            "`final_result` so a hallucinated True gets caught at parse time."
        ),
    )
    final_result: TestExecutionResult = Field(
        ...,
        description=(
            "The last execution's structured result; the evidence backing "
            "the `all_passing` claim above."
        ),
    )

    @model_validator(mode="after")
    def _all_passing_matches_final_result(self) -> GeneratedTest:
        """`all_passing` is a headline claim; `final_result` is the
        evidence. Enforce that they agree at parse time so a future
        prompt change that lets the model set `all_passing=True` while
        the last real execution had failures surfaces immediately."""
        actual_passing = (
            self.final_result.tests_failed == 0
            and not self.final_result.timed_out
            and self.final_result.tests_passed > 0
        )
        if self.all_passing and not actual_passing:
            raise ValueError(
                f"all_passing=True but final_result shows "
                f"{self.final_result.tests_failed} failed, "
                f"{self.final_result.tests_passed} passed, "
                f"timed_out={self.final_result.timed_out}. "
                "Model reported success without evidence."
            )
        if not self.all_passing and actual_passing:
            # Auto-correct rather than raise: an honestly-conservative
            # model saying False when the run passed is a smaller failure
            # than the reverse, and we can trust the underlying evidence.
            self.all_passing = True
        return self
