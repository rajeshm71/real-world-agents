"""Pydantic models for the interview-prep agent's structured output.

Three layers:

- `AnswerFeedback` / `InterviewTurn`: what the LLM actually returns
  each turn (`output_type=InterviewTurn` on the OpenAI Agents SDK
  `Agent`). One turn = one assessment of the previous answer (if any)
  plus one decision about what to do next.
- `TurnRecord`: what the driving loop builds from one `InterviewTurn`
  plus the candidate's answer to it. Never returned by the LLM
  directly -- `topic_index` here is loop-owned bookkeeping, not
  something the model can see or set.
- `TopicSummary` / `InterviewSession`: the closing report, built by
  pure-Python aggregation over the recorded `TurnRecord`s (see
  agent.py's `_build_session_report`) -- no extra LLM call.

Design notes:

- `InterviewTurn.topic` is the model's own free-text label for
  display purposes ONLY. It is never used to decide whether two turns
  are "the same topic" -- that's `TurnRecord.topic_index`, which the
  driving loop assigns deterministically. A model that reuses
  inconsistent topic labels across turns can't corrupt the topic-cap
  bookkeeping this way.
- `requirement_from_jd` is a soft-grounding field: the system prompt
  asks the model to paraphrase an actual JD requirement here, but
  nothing validates it as a verbatim substring of the JD (unlike
  agent #04's `verify_excerpt` tool). A strict validator was
  considered and rejected -- it would be brittle across models/
  providers and isn't required for the core technique to be real.
- `question` is required to be `None` exactly when `action ==
  "conclude"`, and non-null otherwise -- enforced by a model
  validator, same [C5] cross-field-invariant pattern as agent #04's
  `_null_out_unknown_owners`.
- `role_title` is populated by the model on turn 1 only, by prompt
  convention (see prompts/interviewer.txt) -- not enforced by the
  schema. Same precedent as #04 leaving `meeting_topic` to the
  model's judgment rather than validating it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Action = Literal["ask_followup", "ask_new_topic", "conclude"]


class AnswerFeedback(BaseModel):
    """Assessment of the candidate's previous answer. Absent on turn 1
    (there is no previous answer yet)."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "How well the answer addressed the question. Use the FULL "
            "0-100 range, not a narrow band: 0-20 no real answer/off-topic, "
            "21-40 vague/generic, 41-60 adequate but shallow, 61-80 solid "
            "with a specific example, 81-100 exceptional with real "
            "judgment. A genuinely detailed answer with a concrete example "
            "should score 70+, not single digits."
        ),
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What the answer did well. Empty list if genuinely nothing stood out.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="What a stronger answer would have covered. Empty list for a complete answer.",
    )
    improvement_tip: str = Field(
        ...,
        min_length=1,
        description="One concrete, actionable tip for improving this specific answer.",
    )


class InterviewTurn(BaseModel):
    """One turn of the interview: an assessment of the previous answer
    (if any) plus the next thing to do. This is the SDK's
    `output_type` -- one call, one validated turn."""

    action: Action = Field(
        ...,
        description=(
            "'ask_followup': drill deeper on the CURRENT topic (previous "
            "answer was thin). 'ask_new_topic': move to a fresh JD "
            "requirement (previous answer was solid, or this is turn 1). "
            "'conclude': enough distinct requirements have been covered; "
            "end the interview."
        ),
    )
    topic: str = Field(
        ...,
        min_length=1,
        description=(
            "Short human-readable label for the current topic (e.g. "
            "'Distributed systems experience'). Display only -- do not "
            "rely on this being consistent across turns for anything "
            "load-bearing."
        ),
    )
    requirement_from_jd: str = Field(
        ...,
        min_length=1,
        description="The specific JD requirement this question targets, closely paraphrased.",
    )
    question: str | None = Field(
        default=None,
        description="The next question to ask the candidate. None iff action == 'conclude'.",
    )
    feedback_on_previous_answer: AnswerFeedback | None = Field(
        default=None,
        description="Assessment of the answer to the PREVIOUS question. None on turn 1.",
    )
    role_title: str | None = Field(
        default=None,
        description="The role title extracted from the JD. Populate on turn 1 only; leave null on every later turn.",
    )

    @model_validator(mode="after")
    def _question_matches_action(self) -> InterviewTurn:
        """`question` must be present for every action except
        'conclude', and absent for 'conclude'. The SDK's structured-
        output validation enforces field types but not this cross-
        field rule -- same [C5] pattern as #04's schemas.py."""
        if self.action == "conclude" and self.question is not None:
            raise ValueError("question must be null when action == 'conclude'")
        if self.action != "conclude" and self.question is None:
            raise ValueError(f"question is required when action == {self.action!r}")
        return self


class TurnRecord(BaseModel):
    """One full turn: what was asked/assessed, plus what the candidate
    said in response. Built by the driving loop in agent.py -- never
    returned by the LLM directly."""

    topic_index: int = Field(
        ...,
        ge=0,
        description="Loop-owned topic counter. NOT the model's `turn.topic` string -- see module docstring.",
    )
    turn: InterviewTurn
    candidate_answer: str | None = Field(
        default=None,
        description="The candidate's typed answer to `turn.question`. None for the final 'conclude' turn, which has no question to answer.",
    )


class TopicSummary(BaseModel):
    """Aggregated view of one topic across however many turns it took."""

    topic_index: int = Field(..., ge=0)
    topic_label: str = Field(
        ...,
        description=(
            "The label from the turn that established this topic_index "
            "(its first turn), not necessarily its most recent one -- a "
            "later turn's label can be unreliable if the two-tier safety "
            "cap force-converted it (see agent.py's _build_session_report)."
        ),
    )
    requirement_from_jd: str
    questions_asked: int = Field(..., ge=1)
    best_score: int = Field(..., ge=0, le=100)


class InterviewSession(BaseModel):
    """The closing report. Built entirely by pure-Python aggregation
    over the session's `TurnRecord`s (see `_build_session_report`) --
    no extra LLM call."""

    role_title: str | None
    ended_reason: Literal["concluded", "cap_reached"] = Field(
        ...,
        description=(
            "'concluded': the model chose action='conclude'. 'cap_reached': "
            "max_topics was exhausted before the model concluded naturally."
        ),
    )
    topics: list[TopicSummary]
    transcript: list[TurnRecord]
    overall_readiness_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Mean of each topic's best_score.",
    )
    top_improvement_areas: list[str] = Field(
        default_factory=list,
        description="Gaps pulled from the lowest-scoring topic(s).",
    )
