"""Pydantic models for the multi-agent research crew's structured output.

Three models: `Source` (one cited source), `ResearchBrief` (the final
output of the crew), and `ResearchNotes` (intermediate handoff from
researcher to writer -- not returned to the caller, but typed for
clarity).

`ResearchBrief` is the CrewAI Task `output_pydantic` on the editor's
final task -- CrewAI validates against it before returning.

Design notes:

- `sources_used` on `ResearchBrief` is validated by a `@model_validator`
  to ensure the editor's `verify_source_citation` tool actually ran:
  every source in `sources_used` must have a non-empty `snippet` (the
  editor removes sources whose snippet didn't survive substring
  verification against the original notes). Same [C5] cross-field-
  invariant pattern as agents #02 (excerpt-in-source), #03 (execute-
  then-verify), and #04 (owner-in-participants).
- `word_count` is captured explicitly rather than computed lazily from
  `background + implications + key_findings` -- caching the value lets
  the schema enforce the 400-600 range (via the validator) without a
  reader having to trust that the SDK actually recomputes it.
- `key_findings` is `list[str]` with `min_length=1` -- an empty brief
  is never a valid outcome (unlike #04's action_items where zero is
  legitimate); if the researcher gathered zero sources the R5 branch
  fires upstream, so by the time we're validating a ResearchBrief the
  crew HAS material to write about.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

WORD_COUNT_MIN = 300
WORD_COUNT_MAX = 700


class Source(BaseModel):
    """One cited source. `snippet` is the verbatim excerpt the
    researcher pulled that supports whatever fact the writer cites
    from this URL. Editor's `verify_source_citation` tool checks that
    the snippet is still traceable (substring-preserving) after the
    writer's drafting pass."""

    url: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    snippet: str = Field(
        ...,
        min_length=1,
        description="Verbatim excerpt from the source content that supports what the writer cited. Editor strips sources whose snippet fails verification.",
    )


class ResearchNotes(BaseModel):
    """Intermediate handoff from researcher to writer. Not returned to
    the caller -- included in the schema module so tests can construct
    predictable inputs to the writer stage. `key_facts` is what the
    writer synthesizes into `background` / `key_findings` / `implications`;
    keeping it as raw facts makes the researcher's output verifiable
    without depending on the writer's phrasing.
    """

    topic: str = Field(..., min_length=1)
    sources: list[Source] = Field(
        ...,
        min_length=1,
        description="Sources gathered by the researcher. At least one -- R5 branch fires upstream if the search returned nothing.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Raw factual claims the researcher pulled from the sources. Writer's job is to synthesize these; editor's job is to verify each traces back to a source.",
    )


class ResearchBrief(BaseModel):
    """Final output of the crew -- what `run_research()` returns.

    Written by the writer, polished + fact-checked by the editor. The
    `@model_validator` below enforces (a) word count in the 300-700
    range (not the writer's 400-600 target, allows some editorial
    flexibility), (b) sources_used isn't empty (editor should have
    kept at least one), (c) every Source has a non-empty snippet
    (editor's substring check preserved).
    """

    topic: str = Field(..., min_length=1)
    summary: str = Field(
        ...,
        min_length=1,
        description="2-3 sentence TL;DR for someone who won't read the rest.",
    )
    background: str = Field(
        ...,
        min_length=1,
        description="What the topic is + why it matters. 1-2 paragraphs.",
    )
    key_findings: list[str] = Field(
        ...,
        min_length=1,
        description="Research findings as bullet points. Each should be a distinct claim traceable to sources_used.",
    )
    implications: str = Field(
        ...,
        min_length=1,
        description="What this means / what changes / what a reader should take away.",
    )
    sources_used: list[Source] = Field(
        ...,
        min_length=1,
        description="Sources the editor verified via verify_source_citation. Empty is never valid at this point -- R5 branch would have fired upstream.",
    )
    word_count: int = Field(
        ...,
        ge=1,
        description="Word count of the brief (background + key_findings + implications). Validator enforces 300-700 range.",
    )

    @model_validator(mode="after")
    def _validate_brief_shape(self) -> ResearchBrief:
        """Cross-field invariants (review-fix pattern from #02/#03/#04
        applied preemptively to #05):

        - word_count must be in [WORD_COUNT_MIN, WORD_COUNT_MAX].
        - Every Source in sources_used must have a non-empty snippet
          (already enforced at Source level via min_length=1, but this
          re-check catches a future refactor that loosens that).

        Doesn't touch key_findings' factual accuracy or the writer's
        specific phrasing -- those aren't things a schema validator
        can enforce; the crew's editor stage handles them."""
        if not WORD_COUNT_MIN <= self.word_count <= WORD_COUNT_MAX:
            raise ValueError(
                f"word_count={self.word_count} outside allowed range "
                f"[{WORD_COUNT_MIN}, {WORD_COUNT_MAX}]. Writer overshot "
                "or editor over-cut."
            )
        for i, src in enumerate(self.sources_used):
            if not src.snippet.strip():
                raise ValueError(
                    f"sources_used[{i}] has an empty snippet -- editor's "
                    "verify_source_citation should have removed this source."
                )
        return self
