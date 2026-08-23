"""Pydantic models for the meeting-notes agent's structured output.

Two models: `ActionItem` (one commitment extracted from the meeting)
and `MeetingSummary` (whole-meeting analysis with participants,
decisions, and action items).

`MeetingSummary` is used as the OpenAI Agents SDK `output_type=` --
the SDK enforces schema validation on the model's final output; the
agent's tool calls during the ReAct loop are separately typed.

Design notes:

- `priority` is a closed `Literal` (three values) so the model can't
  invent labels like "medium-high" that break downstream sorting.
- `context_excerpt` is required to be a verbatim substring of the
  input notes -- enforced at agent.py time via the `verify_excerpt`
  tool (Pydantic can't see the source notes from inside a field
  validator). Same [C5] pattern as agent #02.
- `owner` and `due_hint` are optional -- the meeting may not name an
  owner or a deadline. Fabricating either is worse than omitting.
- `participants` on the summary comes from the `extract_speakers`
  tool -- keeps the field grounded in text that actually appeared,
  not invented attendees.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Priority = Literal["low", "medium", "high"]


class ActionItem(BaseModel):
    """One commitment extracted from the meeting. `context_excerpt`
    MUST be a verbatim substring of the input notes; enforced at
    agent.py time (see verify_excerpt tool), not by Pydantic."""

    description: str = Field(
        ...,
        min_length=1,
        description="What needs to be done, in plain language. Not a copy of the excerpt -- a distilled action statement (e.g. 'Send updated proposal to legal').",
    )
    owner: str | None = Field(
        default=None,
        description="Meeting participant who owns this item. Must be a name that appears in MeetingSummary.participants; None if the meeting didn't name an owner. Do not fabricate.",
    )
    due_hint: str | None = Field(
        default=None,
        description="Verbatim date/deadline hint from the notes (e.g. 'next Friday', 'end of sprint', '2026-09-15'). None if no deadline was mentioned. Free-text because meeting dates are messy.",
    )
    priority: Priority = Field(
        ...,
        description="Inferred from context: 'high' for stated urgency ('urgent', 'ASAP', 'blocker'), 'low' for 'when possible' / 'eventually', 'medium' by default.",
    )
    context_excerpt: str = Field(
        ...,
        min_length=1,
        description="Verbatim quote from the input notes showing where this item came from. Validated as a substring of the source at agent.py time.",
    )


class MeetingSummary(BaseModel):
    """The full output of one extract_action_items() call. `participants`
    should include everyone who spoke or was named; `action_items` is
    the substantive output; `key_decisions` captures agreed outcomes
    that aren't action items (e.g. 'Decided to postpone launch to Q4').
    """

    meeting_topic: str | None = Field(
        default=None,
        description="Best-effort topic extraction: 'Q3 planning', 'Bug triage', 'Weekly standup', etc. None if not clearly stated in the notes.",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="People who spoke or were named in the meeting. Populated from extract_speakers tool output -- grounded in real text, not invented.",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Commitments made in the meeting. Empty list is a valid outcome -- some meetings genuinely produce discussion without actionable items.",
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        description="Agreed outcomes that aren't action items ('Decided to postpone launch to Q4'). Empty list if the meeting was purely tactical.",
    )
    overall_summary: str = Field(
        ...,
        min_length=1,
        description="2-4 sentence plain-English summary of what the meeting was about and what changed as a result.",
    )

    @model_validator(mode="after")
    def _null_out_unknown_owners(self) -> MeetingSummary:
        """The system prompt tells the model to only use owners that
        appear in `participants`, but the SDK's structured-output
        validation can't enforce that at the field level. Auto-null any
        owner that isn't in the participants list rather than raising --
        forgiving option so a mostly-good extraction still returns
        something useful, just with the bogus owner cleared.

        Same [C5] cross-field-invariant pattern as agents #02 (excerpt-
        in-source) and #03 (execute-then-verify) -- the schema enforces
        what a per-field validator can't.
        """
        known_participants = set(self.participants)
        for item in self.action_items:
            if item.owner is not None and item.owner not in known_participants:
                item.owner = None
        return self
