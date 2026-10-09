"""Pydantic models for the sales outreach agent's structured input and
output.

Three layers:

- `SenderProfile` / `Lead`: inputs the user supplies. Never invented by
  the model -- `SenderProfile.sender_postal_address` exists specifically
  to satisfy CAN-SPAM's physical-address requirement, and `Lead.blurb`
  is the only source of the "specific detail" every drafted email must
  reference (see `DraftedEmail.personalization_note`).
- `LeadQualification` / `DraftedEmail` / `SendEmailResult`: what one
  LLM call (or, for `SendEmailResult`, one real SMTP attempt) produces
  for one lead in phase 1.
- `LeadOutreachRecord` / `OutreachBatchResult`: phase 1's aggregated
  output. `ReplyClassification` / `ReplyOutcome`: phase 2's output,
  invoked separately whenever a reply actually arrives.

Design notes:

- `LeadQualification.qualified` is never trusted as a raw model-
  returned boolean -- agent.py recomputes it from `fit_score` against
  the configured threshold, the same "don't trust the model's own flag"
  move agent #11's `_recommendation_for` makes for its own
  `recommendation` field.
- `DraftedEmail.personalization_note` and the opt-out/postal-address
  requirement are enforced by agent.py's `extra_validate` (substring
  checks against the lead's `blurb` and the sender's required
  boilerplate), not by a Pydantic field validator here -- Pydantic
  can't see the lead's blurb or the sender profile from inside a
  `DraftedEmail` field validator, same reasoning as agent #02's
  excerpt-in-source check living in agent.py, not schemas.py.
- `LeadOutreachRecord.outcome` is a closed 5-value `Literal`, not a
  free-form string, so a reader can enumerate every possible result of
  one lead going through phase 1 without reading agent.py.
- `ReplyClassification.suggested_next_action` is deliberately an
  untyped free-text field, not a `Literal` -- it is advisory only.
  agent.py NEVER acts on it for `unsubscribe`/`not_interested`; the
  code-owned hard override always wins regardless of what this field
  says. See `ReplyOutcome.override_applied`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SenderProfile(BaseModel):
    """Who the outreach is FROM. Required input, never invented by the
    model. `sender_postal_address` exists specifically to satisfy
    CAN-SPAM's physical-address requirement: this agent guarantees the
    *mechanism* (every draft includes it), the user is responsible for
    the accuracy of what they put in it."""

    sender_name: str = Field(..., min_length=1)
    sender_company: str = Field(..., min_length=1)
    reply_to_note: str = Field(
        ...,
        min_length=1,
        description="How a recipient actually opts out, e.g. 'reply to "
        "this email' -- matches the opt-out line drafted into every "
        "email body.",
    )
    sender_postal_address: str = Field(..., min_length=1)


class Lead(BaseModel):
    """One prospect to triage in phase 1. `lead_id` only needs to be
    unique within a single batch -- phase 2 looks leads up by
    `contact_email` for cross-batch checks (unsubscribe durability),
    not by `lead_id`, since `lead_id` isn't guaranteed stable across
    separately-prepared `leads.json` files for the same person."""

    lead_id: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    contact_name: str = Field(..., min_length=1)
    contact_title: str = Field(..., min_length=1)
    contact_email: str
    blurb: str = Field(
        ...,
        min_length=1,
        description="Free-text specifics about this company/role -- the "
        "personalization fuel. A generic blurb produces a generic email; "
        "this agent can't invent specifics that aren't here.",
    )

    @field_validator("contact_email")
    @classmethod
    def _validate_email_shape(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError(
                f"contact_email {v!r} doesn't look like a valid email "
                "address. Fix the lead's contact_email before submitting "
                "this batch."
            )
        return v


class LeadQualification(BaseModel):
    """Output of the qualify node for one lead."""

    lead_id: str
    fit_score: int = Field(..., ge=0, le=100)
    reasoning: str = Field(..., min_length=1)
    qualified: bool = Field(
        ...,
        description="Recomputed in agent.py from fit_score vs the "
        "configured threshold -- never trusted as the model's own "
        "boolean (see module docstring).",
    )


class DraftedEmail(BaseModel):
    """Output of the draft node for one lead (or one follow-up in
    phase 2)."""

    lead_id: str
    subject: str = Field(..., min_length=1)
    body: str = Field(
        ...,
        min_length=1,
        description="Must include a plainly worded opt-out line and the "
        "sender's postal address (checked in agent.py's extra_validate, "
        "a CAN-SPAM-driven requirement) in addition to the referenced "
        "personalization detail.",
    )
    personalization_note: str = Field(
        ...,
        min_length=1,
        description="Which specific detail from the lead's blurb this "
        "email uses. Must be a verbatim-ish substring also present in "
        "body -- checked in agent.py, not here.",
    )


class SendEmailResult(BaseModel):
    """The real-world outcome of one SMTP send attempt. Never raised as
    an exception for an expected failure (auth rejected, connection
    refused, timeout) -- always returned as this structured result,
    mirroring agent #08's TestExecutionResult shape for its own
    real-world side effect."""

    sent: bool
    error: str | None = Field(
        default=None,
        description="None iff sent=True, or iff this was a dry run "
        "(dry_run=True implies sent=False, error=None -- a true no-op, "
        "not a failure).",
    )
    dry_run: bool = False


class LeadOutreachRecord(BaseModel):
    """One lead's full result from a phase-1 batch run."""

    lead: Lead
    qualification: LeadQualification
    drafted_email: DraftedEmail | None = Field(
        default=None, description="None when outcome is a skip -- no draft was ever attempted."
    )
    send_result: SendEmailResult | None = Field(
        default=None, description="None when outcome is a skip -- no send was ever attempted."
    )
    outcome: Literal[
        "sent",
        "dry_run_drafted",
        "skipped_not_qualified",
        "skipped_previously_unsubscribed",
        "send_failed",
    ]


class OutreachBatchResult(BaseModel):
    """The full output of one `run_outreach_batch()` call."""

    records: list[LeadOutreachRecord]
    run_meta: dict = Field(default_factory=dict)


class ReplyClassification(BaseModel):
    """Output of the classify_reply node in phase 2."""

    lead_id: str
    intent: Literal[
        "interested", "not_interested", "objection", "out_of_office", "unsubscribe"
    ]
    reasoning: str = Field(..., min_length=1)
    suggested_next_action: str = Field(
        ...,
        min_length=1,
        description="Advisory only. NEVER trusted for intent in "
        "(unsubscribe, not_interested) -- the code-owned hard override "
        "in agent.py always wins regardless of what this field says.",
    )
    snooze_until: str | None = Field(
        default=None,
        description="Extracted follow-up-after date for an "
        "out_of_office reply, if one was mentioned. None otherwise.",
    )


class ReplyOutcome(BaseModel):
    """The full output of one `process_reply()` call."""

    lead_id: str
    classification: ReplyClassification
    final_status: Literal["dead", "snoozed", "followup_drafted", "followup_sent"]
    followup_email: DraftedEmail | None = None
    send_result: SendEmailResult | None = None
    override_applied: bool = Field(
        ...,
        description="True only on the code-owned unsubscribe/"
        "not_interested override path -- the field the safety test "
        "asserts on, proving the code overrides the model rather than "
        "merely trusting it.",
    )
