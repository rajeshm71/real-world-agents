"""Pydantic models for the resume screener.

Five types travel the pipeline:

- `ResumeInput`:        loader output, screener input. Filename-derived
                        `resume_id`, human-formatted `candidate_name`,
                        extracted `resume_text`.
- `EvidenceExcerpt`:    one verbatim quote from a resume. Substring
                        check happens on `CandidateScorecard`.
- `DimensionScore`:     one of three scoring dimensions (skills_match,
                        experience_match, culture_signal) with 0-100
                        score, rationale, and evidence.
- `CandidateScorecard`: full per-candidate result. Cross-field
                        validators enforce evidence honesty (verbatim
                        substring of `resume_text`), exactly three
                        dimensions, overall bounded near the dimension
                        mean, recommendation consistent with overall.
- `ScreeningResult`:    batch return. Ranked ids must be a permutation
                        of the scorecards' ids in score-descending
                        order (ties broken by id ascending).
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DIMENSION_NAMES: tuple[str, str, str] = (
    "skills_match",
    "experience_match",
    "culture_signal",
)

_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    """Collapse any run of whitespace to a single space. Used for the
    verbatim-substring check so a PDF's ragged whitespace (line breaks
    injected mid-paragraph, tabs from column extraction) doesn't
    invalidate an otherwise-accurate quote."""
    return _WS_RE.sub(" ", s).strip()


def _substring_of_normalized(needle: str, haystack: str) -> bool:
    """Whitespace-normalized substring check. Same helper shape as
    #10's `_substring_of_normalized`; kept private per agent so
    changes to the tolerance policy here don't ripple across the
    catalog."""
    return _normalize_whitespace(needle) in _normalize_whitespace(haystack)


class ResumeInput(BaseModel):
    """Loader output = screener input.

    `resume_id` is a stable slug (lowercased filename stem). The
    caller can override to disambiguate two resumes with the same
    filename. `candidate_name` is a display string derived from the
    stem (`alice_smith.pdf` -> `Alice Smith`).
    """

    resume_id: str = Field(..., min_length=1)
    candidate_name: str = Field(..., min_length=1)
    resume_text: str = Field(..., min_length=1)


class EvidenceExcerpt(BaseModel):
    """One verbatim quote from a resume backing a dimension score."""

    quoted_text: str = Field(..., min_length=1)


class DimensionScore(BaseModel):
    """One dimension of the scorecard.

    `evidence` is required whenever `score >= 40` -- the model must
    ground even a mediocre positive score in the resume text. A score
    below 40 means "not enough evidence found," which the rubric
    treats as no-hire signal on that dimension.
    """

    name: Literal["skills_match", "experience_match", "culture_signal"]
    score: int = Field(..., ge=0, le=100)
    rationale: str = Field(..., min_length=1)
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_required_for_positive_score(self) -> DimensionScore:
        if self.score >= 40 and not self.evidence:
            raise ValueError(
                f"dimension {self.name!r} scored {self.score} but has no "
                "evidence; every score >= 40 must cite at least one "
                "verbatim excerpt from the resume."
            )
        return self


class CandidateScorecard(BaseModel):
    """Full per-candidate result.

    The LLM only returns the scoring fields (`dimensions`,
    `overall_score`, `overall_rationale`, `recommendation`); the
    caller injects `resume_id`, `candidate_name`, and `resume_text`
    from what it already knows. Same "model returns partial, caller
    fills the rest" pattern as #10's `retrieved_chunks`.
    """

    resume_id: str = Field(..., min_length=1)
    candidate_name: str = Field(..., min_length=1)
    resume_text: str = Field(..., min_length=1)
    dimensions: list[DimensionScore]
    overall_score: int = Field(..., ge=0, le=100)
    overall_rationale: str = Field(..., min_length=1)
    recommendation: Literal["strong_yes", "yes", "borderline", "no"]

    @model_validator(mode="after")
    def _dimensions_are_the_canonical_three(self) -> CandidateScorecard:
        names = [d.name for d in self.dimensions]
        if sorted(names) != sorted(DIMENSION_NAMES):
            raise ValueError(
                f"dimensions must be exactly the canonical set "
                f"{sorted(DIMENSION_NAMES)!r}; got {sorted(names)!r}."
            )
        return self

    @model_validator(mode="after")
    def _evidence_is_verbatim_substring_of_resume(self) -> CandidateScorecard:
        for dim in self.dimensions:
            for excerpt in dim.evidence:
                if not _substring_of_normalized(excerpt.quoted_text, self.resume_text):
                    raise ValueError(
                        f"dimension {dim.name!r} cites quoted_text that is "
                        f"not a verbatim substring of the resume "
                        f"(whitespace-normalized): {excerpt.quoted_text[:80]!r}"
                    )
        return self

    @model_validator(mode="after")
    def _overall_near_dimension_mean(self) -> CandidateScorecard:
        mean = sum(d.score for d in self.dimensions) / len(self.dimensions)
        if abs(self.overall_score - mean) > 10:
            raise ValueError(
                f"overall_score ({self.overall_score}) drifts more than 10 "
                f"from the dimension mean ({mean:.1f}); model must weight "
                "dimensions, not invent an overall."
            )
        return self

    @model_validator(mode="after")
    def _recommendation_matches_rubric(self) -> CandidateScorecard:
        expected = _recommendation_for(self.overall_score)
        if self.recommendation != expected:
            raise ValueError(
                f"recommendation {self.recommendation!r} contradicts "
                f"overall_score {self.overall_score} (rubric expects "
                f"{expected!r}); rubric: >=80 strong_yes, 60-79 yes, "
                "40-59 borderline, <40 no."
            )
        return self


def _recommendation_for(
    overall_score: int,
) -> Literal["strong_yes", "yes", "borderline", "no"]:
    """Rubric mapping used both by the scorecard validator and by the
    mock-mode generator so they stay in lockstep."""
    if overall_score >= 80:
        return "strong_yes"
    if overall_score >= 60:
        return "yes"
    if overall_score >= 40:
        return "borderline"
    return "no"


class ScreeningResult(BaseModel):
    """Batch return.

    `run_meta` captures provider/model/elapsed for observability; the
    CLI writes the full ScreeningResult to `last_run.json` so a real
    HR reviewer can audit what the model was told and how it scored.
    """

    job_description: str = Field(..., min_length=1)
    scorecards: list[CandidateScorecard]
    ranked_ids: list[str]
    run_meta: dict[str, Any]

    @model_validator(mode="after")
    def _ranked_is_permutation_of_scorecard_ids(self) -> ScreeningResult:
        scorecard_ids = [s.resume_id for s in self.scorecards]
        if sorted(self.ranked_ids) != sorted(scorecard_ids):
            raise ValueError(
                f"ranked_ids must be a permutation of the scorecards' "
                f"resume_ids. scorecards: {sorted(scorecard_ids)!r}; "
                f"ranked: {sorted(self.ranked_ids)!r}."
            )
        return self

    @model_validator(mode="after")
    def _rank_order_is_score_descending(self) -> ScreeningResult:
        by_id = {s.resume_id: s for s in self.scorecards}
        ordered = [by_id[rid] for rid in self.ranked_ids]
        for prev, curr in pairwise(ordered):
            if prev.overall_score < curr.overall_score:
                raise ValueError(
                    f"ranked_ids must be non-increasing by overall_score; "
                    f"{prev.resume_id!r} ({prev.overall_score}) precedes "
                    f"{curr.resume_id!r} ({curr.overall_score})."
                )
            if (
                prev.overall_score == curr.overall_score
                and prev.resume_id > curr.resume_id
            ):
                raise ValueError(
                    f"ties on overall_score must break by resume_id "
                    f"ascending for determinism; got {prev.resume_id!r} "
                    f"before {curr.resume_id!r} at equal score "
                    f"{prev.overall_score}."
                )
        return self
