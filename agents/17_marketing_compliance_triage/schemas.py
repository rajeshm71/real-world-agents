"""Pydantic models for the marketing compliance triage agent's structured
output.

Three layers:

- `LensClassification`: the classify node's output. `active_lenses`
  drives the graph's conditional fan-out directly (see agent.py's
  `_route_after_classify`) -- it is never re-derived from `reasoning`.
- `LensFinding`: one specialist reviewer node's output for ONE lens.
- `ComplianceReport`: the final aggregated output, built either by the
  `aggregate` node (when at least one lens is active) or directly in
  Python with no LLM call (when zero lenses are active -- see
  agent.py's `_aggregate`).

Design notes:

- `flagged_phrases` gets a lighter-touch verbatim-substring check than
  agent #02's `excerpt` field (checked in agent.py's retry loop, not
  here -- Pydantic can't see `copy_text` from inside a field
  validator). Marketing copy is short (a few sentences to a paragraph)
  compared to a whole contract, so this agent checks a list of phrases
  per finding rather than enforcing a single excerpt field.
- `overall_risk_level` has a FLOOR of the worst individual finding's
  severity (enforced in agent.py's `_aggregate` extra-validation, not
  here) but can be explicitly escalated above it -- several medium
  findings across different lenses can compound into a high overall
  risk even if no single finding is high on its own.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Lens = Literal[
    "health_wellness",
    "financial_earnings",
    "childrens_privacy",
    "environmental",
]

Severity = Literal["low", "medium", "high"]


class LensClassification(BaseModel):
    """Output of the classify node."""

    active_lenses: list[Lens] = Field(
        default_factory=list,
        description=(
            "Which of the 4 fixed regulatory lenses apply to this copy. "
            "An empty list is a valid, real result -- plenty of marketing "
            "copy (e.g. a pure brand-awareness ad with no claims) "
            "triggers none of them. Do not force-fit a lens that doesn't "
            "genuinely apply."
        ),
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "1-3 sentence plain-English justification for which lenses "
            "were selected (and, implicitly, why the others weren't). "
            "Shown in the UI so a human reviewer can sanity-check the "
            "classifier's judgment call, not just trust it blindly."
        ),
    )


class LensFinding(BaseModel):
    """One specialist reviewer node's output for ONE lens. A single
    lens can flag multiple phrases -- flagged_phrases is a list, not a
    single excerpt, because one pass over an ad often finds 2-3
    separate problematic phrases at once."""

    lens: Lens
    flagged_phrases: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim phrases from the copy that triggered this lens's "
            "concern. Checked as substrings of the source copy in "
            "agent.py's retry loop (whitespace-normalized), not "
            "enforced by Pydantic here -- Pydantic can't see the "
            "source text from inside a field validator."
        ),
    )
    concern: str = Field(
        ...,
        min_length=1,
        description=(
            "Plain-English statement of the regulatory risk, e.g. "
            "'unsubstantiated efficacy claim under FTC health-claim "
            "substantiation doctrine'."
        ),
    )
    suggested_fix: str = Field(
        ...,
        min_length=1,
        description=(
            "Concrete rewording or disclosure addition a marketing team "
            "could apply directly -- not generic legal advice."
        ),
    )
    severity: Severity


class ComplianceReport(BaseModel):
    """The final aggregated output. Built by agent.py's `_aggregate`
    node -- either via an LLM call (when `active_lenses` is non-empty)
    or directly in Python with no LLM call at all (when it's empty)."""

    active_lenses: list[Lens] = Field(default_factory=list)
    findings: list[LensFinding] = Field(
        default_factory=list,
        description=(
            "All findings across all active lenses, in the order the "
            "reducer merged them. Parallel-branch completion order is "
            "not guaranteed stable across runs -- see agent.py's module "
            "docstring for why this is an acceptable, documented "
            "non-determinism (it affects list ORDER only, never which "
            "findings are present)."
        ),
    )
    overall_risk_level: Severity
    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "2-4 sentence plain-English summary across ALL active "
            "lenses' findings. When active_lenses is empty, this is a "
            "clean 'no regulatory concerns identified across the 4 "
            "lenses this agent checks' statement, not an empty string "
            "and not an error."
        ),
    )
