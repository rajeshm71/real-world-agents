"""Pydantic models for the from-scratch agent loop.

Four types travel the loop:

- `ToolCall`:   one tool call decoded from the LLM's response.
- `ToolResult`: one tool's outcome, stringified for the LLM's next turn.
- `AgentStep`:  one iteration of the loop (assistant preamble + calls +
                results).
- `AgentTrace`: the full return of `ask()` -- question, final answer,
                the ordered steps, and how the loop terminated.

Cross-field validators on `AgentTrace` enforce the loop's contract:
a `final_answer` termination must actually have an answer; an `error`
termination must carry a reason; iteration count must line up with
the recorded steps.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """One tool call as decoded from the LLM's response.

    `call_id` echoes OpenAI's `tool_call_id` so the tool result can be
    routed back to the right assistant message on the next turn.
    """

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any]
    call_id: str = Field(..., min_length=1)


class ToolResult(BaseModel):
    """One tool's outcome.

    `content` is always a string -- structured returns are pre-stringified
    by the dispatcher because the LLM only ever sees text in the `tool`
    message role. `error` is set when the tool raised or the arguments
    didn't parse; the loop keeps going, letting the model see the
    failure and correct course.
    """

    call_id: str = Field(..., min_length=1)
    content: str
    error: str | None = None


class AgentStep(BaseModel):
    """One iteration of the loop.

    `assistant_message` may be None (the model chose to emit only tool
    calls) OR a string (a preamble like "Let me check..." alongside the
    tool calls). Both cases show up in real traces.
    """

    iteration: int = Field(..., ge=1)
    assistant_message: str | None
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]

    @model_validator(mode="after")
    def _results_match_calls(self) -> AgentStep:
        if len(self.tool_calls) != len(self.tool_results):
            raise ValueError(
                f"AgentStep has {len(self.tool_calls)} tool_calls but "
                f"{len(self.tool_results)} tool_results; the dispatcher "
                "must produce exactly one result per call, in order."
            )
        for call, result in zip(self.tool_calls, self.tool_results, strict=True):
            if call.call_id != result.call_id:
                raise ValueError(
                    f"AgentStep tool_calls / tool_results out of order: "
                    f"call_id {call.call_id!r} paired with result call_id "
                    f"{result.call_id!r}."
                )
        return self


class AgentTrace(BaseModel):
    """The full return of `ask()`.

    `steps` records each iteration that made a tool call. A step whose
    assistant response was a plain final answer (no tool calls) does
    NOT appear here -- that response IS `final_answer`, and
    `iterations_used` is the count of LLM calls including that final
    one.
    """

    question: str = Field(..., min_length=1)
    final_answer: str
    steps: list[AgentStep]
    terminated_by: Literal["final_answer", "max_iterations", "error"]
    iterations_used: int = Field(..., ge=0)
    error_reason: str | None = None

    @model_validator(mode="after")
    def _final_answer_consistent(self) -> AgentTrace:
        if self.terminated_by == "final_answer":
            if not self.final_answer.strip():
                raise ValueError(
                    "terminated_by='final_answer' requires a non-empty final_answer."
                )
            if self.error_reason is not None:
                raise ValueError(
                    "terminated_by='final_answer' must not carry an error_reason."
                )
        return self

    @model_validator(mode="after")
    def _error_carries_reason(self) -> AgentTrace:
        if self.terminated_by == "error" and not (self.error_reason or "").strip():
            raise ValueError(
                "terminated_by='error' requires a non-empty error_reason."
            )
        return self

    @model_validator(mode="after")
    def _iterations_match_steps(self) -> AgentTrace:
        # A max-iterations termination means every iteration produced tool
        # calls, so steps and iterations agree exactly. A final_answer or
        # error termination may add one extra LLM call (the final answer,
        # or the call that errored) beyond the recorded tool-call steps.
        if self.terminated_by == "max_iterations":
            if self.iterations_used != len(self.steps):
                raise ValueError(
                    f"terminated_by='max_iterations' requires "
                    f"iterations_used ({self.iterations_used}) == "
                    f"len(steps) ({len(self.steps)})."
                )
        elif self.iterations_used < len(self.steps):
            raise ValueError(
                f"iterations_used ({self.iterations_used}) cannot be less "
                f"than len(steps) ({len(self.steps)})."
            )
        return self
