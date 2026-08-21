"""Pydantic models for the email-triage agent's structured output.

Three types: `EmailTriage` (the agent's typed output), `TriageDeps`
(dependency-injection payload for tools via `RunContext[TriageDeps]`),
and the Category/Priority/Action Literal aliases.

`EmailTriage` is used as PydanticAI Agent's `output_type=` --
PydanticAI validates the model's structured output against it. The
`@model_validator` below enforces a cross-field invariant: only
respond_now / respond_later actions may have a non-None
`suggested_reply` (H1-option-b pattern from agent #04).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal[
    "work",
    "personal",
    "newsletter",
    "spam",
    "important",
    "notification",
    "other",
]
Priority = Literal["urgent", "high", "medium", "low"]
Action = Literal[
    "respond_now",
    "respond_later",
    "delegate",
    "archive",
    "ignore",
]

_RESPOND_ACTIONS = frozenset({"respond_now", "respond_later"})


@dataclass
class TriageDeps:
    """User context injected into every triage. PydanticAI tools
    receive this via `RunContext[TriageDeps]` and use it to ground
    decisions in real user context rather than guesses."""

    known_contacts: list[str] = field(default_factory=list)
    important_domains: list[str] = field(default_factory=list)


class EmailTriage(BaseModel):
    """Structured triage decision for one email. Cross-field
    invariant (respond-action <-> suggested_reply) enforced by the
    validator below."""

    category: Category
    priority: Priority
    action: Action
    reasoning: str = Field(
        ...,
        min_length=1,
        description="2-3 sentence justification for the triage decision. Not a restatement of the email; an actual explanation.",
    )
    key_snippet: str = Field(
        ...,
        min_length=1,
        description="Verbatim excerpt from the email body supporting the triage. Validated as a substring of the body at agent.py time (verify_snippet tool -- [C5] pattern).",
    )
    suggested_reply: str | None = Field(
        default=None,
        description="Draft reply text. Only meaningful when action is 'respond_now' or 'respond_later' -- auto-cleared to None otherwise by the validator.",
    )

    @model_validator(mode="after")
    def _validate_reply_action_consistency(self) -> EmailTriage:
        """Auto-null `suggested_reply` for non-respond actions.
        Same H1-option-b (forgiving) pattern as #04's MeetingSummary
        owner-in-participants check -- a mostly-good triage where the
        model produced a reply for a non-respond action still returns
        something useful, just with the mismatched field cleared."""
        if self.action not in _RESPOND_ACTIONS and self.suggested_reply is not None:
            self.suggested_reply = None
        return self
