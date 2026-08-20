"""Pydantic models for the contract reviewer's structured output.

Two models: `ContractFlag` (one flagged clause + its risk explanation) and
`ContractReview` (whole-contract analysis). Together they're the Pydantic
target the hand-rolled JSON-validate-retry loop in `agent.py` validates
against -- if the model returns JSON that fails these constraints, the
loop re-prompts with the validation error.

Design notes:

- `category` is a closed `Literal` (not an open `str`) so the model can't
  invent a private category label like `"weird_thing"` -- Pydantic rejects
  it, the retry loop feeds the error back. Ten specific categories were
  chosen to cover the clauses a non-lawyer freelancer / small-business
  operator actually needs to know about before signing (auto-renewal,
  unlimited liability, one-sided termination, etc.), plus `"other"` for
  the honest cases where the model spots something risky that doesn't
  fit the taxonomy.
- `page_number` (int, not free-text) is populated from `--- PAGE N ---`
  markers `agent.py` injects into the prompt. A verifiable field the user
  can actually navigate to, not a hallucination-prone free-text
  "section 4.2(b)" hint (which the model would confidently fabricate).
- `excerpt` is required to be a substring of the contract source text at
  validation time -- that check happens in `agent.py`'s retry loop
  (Pydantic can't see the source text from inside a field validator),
  not here. The docstring notes the invariant so a reader knows why the
  check lives in agent.py, not schemas.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FlagCategory = Literal[
    "auto_renewal",
    "unlimited_liability",
    "unilateral_termination",
    "arbitration",
    "non_compete",
    "ip_assignment",
    "indemnification",
    "exclusivity",
    "governing_law",
    "confidentiality",
    "other",
]

Severity = Literal["low", "medium", "high"]


class ContractFlag(BaseModel):
    """One flagged clause in the contract. `excerpt` MUST be a verbatim
    substring of the contract source text -- enforced by agent.py's retry
    loop, not by Pydantic (Pydantic can't see the source from inside a
    field validator)."""

    category: FlagCategory
    severity: Severity
    excerpt: str = Field(
        ...,
        min_length=1,
        description="Verbatim quote from the contract, so the user can locate the clause themselves. Validated as a substring of the source text at agent.py:review_contract time.",
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="Page number the excerpt appears on. Populated from '--- PAGE N ---' markers that agent.py injects into the prompt. None only if the source input was plain text (no page structure).",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Plain-English risk statement for a non-lawyer. Not legal jargon; not a restatement of the excerpt.",
    )
    recommendation: str | None = Field(
        default=None,
        description="Optional: what the user might push back on before signing. Left None when no obvious counter-move applies.",
    )


class ContractReview(BaseModel):
    """Whole-contract analysis. `contract_type` and `parties` are best-
    effort extractions from the document (may be None/empty if the model
    can't determine them); `flags` is the substantive output."""

    contract_type: str | None = Field(
        default=None,
        description="Best-effort contract-type label the model detects first (NDA / MSA / SaaS ToS / employment / vendor SOW / etc.). Fed back into the model's own scrutiny via the prompt's chain-of-thought instructions -- see prompts/review.txt.",
    )
    parties: list[str] = Field(
        default_factory=list,
        description="Contracting parties (company names, individual names). Empty list if the model can't identify them.",
    )
    flags: list[ContractFlag] = Field(
        default_factory=list,
        description="Flagged clauses, in the order the model surfaces them. An empty list is a real, valid result -- some contracts genuinely have no high-risk clauses for a non-lawyer to worry about.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="2-4 sentence plain-English summary of what the contract does and its overall risk shape.",
    )
    overall_risk: Severity
