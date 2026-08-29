"""Pydantic models for the fact-checker.

Five types travel the load -> extract -> verify -> report pipeline:

- `InputSource`:      what the caller passed (text, transcript file,
                      audio file, or YouTube URL).
- `FactualClaim`:     one extracted verifiable claim.
- `EvidenceSnippet`:  one verbatim excerpt from a search result.
- `ClaimVerdict`:     verdict + confidence + explanation + evidence
                      for a single claim.
- `FactCheckReport`:  full pipeline return.

Cross-field validators enforce (a) InputSource's kind + payload
consistency, (b) verdict / evidence / confidence consistency (no
supported-without-evidence, no unverifiable-with-evidence, no
low-confidence + strong verdict), and (c) claim text is a verbatim
substring of the source text (no invented claims).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _substring_of_normalized(needle: str, haystack: str) -> bool:
    if not needle.strip():
        return False
    return _normalize_whitespace(needle) in _normalize_whitespace(haystack)


class InputSource(BaseModel):
    kind: Literal["text", "transcript_file", "audio", "youtube"]
    path: Path | None = None
    url: str | None = None
    raw_text: str | None = None

    @model_validator(mode="after")
    def _kind_and_payload_agree(self) -> InputSource:
        provided = [
            name
            for name, val in (("path", self.path), ("url", self.url), ("raw_text", self.raw_text))
            if val is not None
        ]
        if len(provided) != 1:
            raise ValueError(
                f"exactly one of path/url/raw_text must be set on InputSource; "
                f"got {provided!r}."
            )
        expected = {
            "text": "raw_text",
            "transcript_file": "path",
            "audio": "path",
            "youtube": "url",
        }[self.kind]
        if provided[0] != expected:
            raise ValueError(
                f"kind={self.kind!r} expects {expected!r} to be set, "
                f"but {provided[0]!r} was provided instead."
            )
        return self


class FactualClaim(BaseModel):
    claim_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=10)
    claim_type: Literal["statistic", "date", "quote", "event", "attribution", "other"]
    entities: list[str] = Field(default_factory=list, max_length=6)
    approx_char_offset: int | None = None


class EvidenceSnippet(BaseModel):
    source_url: str = Field(..., min_length=1)
    source_title: str = Field(..., min_length=1)
    quoted_text: str = Field(..., min_length=1)
    search_provider: Literal["tavily", "brave", "ddg"]


class ClaimVerdict(BaseModel):
    claim: FactualClaim
    verdict: Literal["supported", "contradicted", "unclear", "unverifiable"]
    confidence: Literal["high", "medium", "low"]
    explanation: str = Field(..., min_length=1)
    evidence: list[EvidenceSnippet] = Field(default_factory=list, max_length=5)
    # How the verdict was reached. "model_knowledge" is the fast-path
    # taken when a claim is answerable from well-established knowledge
    # (basic math, universally-known facts) and skipping the web
    # search saves real cost. "web_search" is the default.
    verification_method: Literal["model_knowledge", "web_search"] = "web_search"

    @model_validator(mode="after")
    def _strong_verdict_evidence_matches_method(self) -> ClaimVerdict:
        if self.verdict in ("supported", "contradicted"):
            if self.verification_method == "web_search" and not self.evidence:
                raise ValueError(
                    f"verdict={self.verdict!r} via web_search requires at "
                    "least one evidence snippet; a strong verdict without "
                    "evidence is a hallucination."
                )
            if self.verification_method == "model_knowledge" and self.evidence:
                raise ValueError(
                    "verification_method='model_knowledge' means no search "
                    "was run, so evidence must be empty."
                )
            if self.confidence == "low":
                raise ValueError(
                    f"verdict={self.verdict!r} + confidence='low' is "
                    "contradictory; downgrade the verdict to 'unclear' if "
                    "the evidence is weak."
                )
        return self

    @model_validator(mode="after")
    def _unverifiable_has_no_evidence(self) -> ClaimVerdict:
        if self.verdict == "unverifiable" and self.evidence:
            raise ValueError(
                "verdict='unverifiable' means nothing was found; evidence "
                "must be empty."
            )
        return self

    @model_validator(mode="after")
    def _model_knowledge_only_for_strong_verdicts(self) -> ClaimVerdict:
        # The fast path is intentionally limited to supported/contradicted:
        # if the model can't confidently take a side from its own
        # knowledge, we WANT it to fall through to web search.
        if self.verification_method == "model_knowledge" and self.verdict in (
            "unclear",
            "unverifiable",
        ):
            raise ValueError(
                f"verification_method='model_knowledge' with verdict="
                f"{self.verdict!r} is nonsensical: if the model is unsure, "
                "the caller should have routed to web search instead."
            )
        return self


class FactCheckReport(BaseModel):
    source: InputSource
    source_text: str = Field(..., min_length=1)
    claims: list[FactualClaim]
    verdicts: list[ClaimVerdict]
    summary: dict[str, int] = Field(default_factory=dict)
    run_meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_verdict_per_claim_matched_by_id(self) -> FactCheckReport:
        claim_ids = [c.claim_id for c in self.claims]
        verdict_ids = [v.claim.claim_id for v in self.verdicts]
        if claim_ids != verdict_ids:
            raise ValueError(
                f"verdicts must be parallel to claims by claim_id; claims: "
                f"{claim_ids!r}; verdicts: {verdict_ids!r}."
            )
        return self

    @model_validator(mode="after")
    def _claims_are_substrings_of_source(self) -> FactCheckReport:
        for c in self.claims:
            if not _substring_of_normalized(c.text, self.source_text):
                raise ValueError(
                    f"claim {c.claim_id!r} text is not a verbatim substring of "
                    f"source_text (whitespace-normalized): {c.text[:80]!r}. "
                    "The extract stage should have already dropped invented "
                    "claims."
                )
        return self
