"""Pydantic schemas for the policy Q&A RAG agent.

Four models:
- `Chunk`: one atomic passage from a source document, with a stable
  ID (`filename#chunk-N`) callers use for citations.
- `Citation`: one claim in an answer, backed by verbatim `quoted_text`
  from a specific chunk. Validator cross-checks the substring at
  parse time.
- `IngestionStats`: what a completed `ingest_corpus()` call reports
  back so the caller can eyeball whether the run made sense.
- `PolicyAnswer`: the final answer + citations + retrieval audit
  trail. Three cross-field validators enforce citation honesty:
  1. Every citation's chunk_id must appear in retrieved_chunks (no
     phantom citations to chunks the retriever never returned).
  2. If confidence is "low", unanswered_reason must be non-empty
     (model must explain why it can't answer).
  3. High/medium confidence answers must have at least one citation
     (an answer without evidence is a hallucination).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """One atomic passage from a source document."""

    chunk_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable ID for this chunk, format `<source_filename>#chunk-<N>` "
            "(e.g. `benefits.md#chunk-3`). Callers cite by this ID."
        ),
    )
    source_path: str = Field(
        ..., description="Path to the source file, relative to the corpus dir."
    )
    text: str = Field(
        ..., min_length=1, description="The chunk's text content, verbatim."
    )
    section: str | None = Field(
        None,
        description=(
            "Nearest Markdown heading (`# H1` or `## H2`) preceding this "
            "chunk, if any. Helps a reader place the citation in context."
        ),
    )


class Citation(BaseModel):
    """One factual claim in an answer, backed by a specific chunk."""

    chunk_id: str = Field(
        ...,
        min_length=1,
        description=(
            "The chunk_id being cited. Must match a chunk in "
            "PolicyAnswer.retrieved_chunks -- enforced by the outer "
            "PolicyAnswer validator."
        ),
    )
    quoted_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Verbatim substring of the cited chunk's text. Not "
            "paraphrased -- the caller cross-checks this at parse time "
            "to catch a model that invents quotes."
        ),
    )


class IngestionStats(BaseModel):
    """What ingest_corpus() reports after a successful run."""

    total_files: int = Field(..., ge=0, description="Source files processed.")
    total_chunks: int = Field(
        ..., ge=0, description="Chunks written to the vector store."
    )
    index_size_bytes: int = Field(
        ..., ge=0, description="Size of the on-disk index (0 for in-memory FAISS)."
    )
    embedding_model: str = Field(..., description="Name of the embedding model used.")
    index_path: str = Field(..., description="Where the index was written.")
    vector_store_backend: Literal["sqlite-vec", "faiss"] = Field(
        ...,
        description=(
            "Which backend actually ended up storing the vectors. "
            "sqlite-vec is preferred; faiss is the fallback when "
            "sqlite-vec's extension load fails."
        ),
    )


class PolicyAnswer(BaseModel):
    """The RAG agent's final structured answer with citations."""

    question: str = Field(..., min_length=1, description="The user's question, echoed back.")
    answer: str = Field(
        ...,
        min_length=1,
        description=(
            "Plain-English answer. May contain inline `[chunk_id]` "
            "markers matching entries in `citations`."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Every factual claim in `answer` should map to a Citation "
            "here. Non-empty for high/medium confidence (enforced by "
            "validator); may be empty when confidence=low."
        ),
    )
    retrieved_chunks: list[Chunk] = Field(
        default_factory=list,
        description=(
            "The top-K chunks the retriever returned, in ranked order. "
            "Audit trail: lets a caller inspect what the model was "
            "shown before it wrote the answer."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "Model's own confidence that the retrieved passages actually "
            "answer the question. `low` means the answer isn't reliably "
            "supported by the corpus."
        ),
    )
    unanswered_reason: str | None = Field(
        None,
        description=(
            "When confidence=low, why the retrieved passages don't answer "
            "the question. Required for low confidence (enforced by "
            "validator); ignored otherwise."
        ),
    )

    @model_validator(mode="after")
    def _citations_reference_retrieved_chunks(self) -> PolicyAnswer:
        """Every citation must cite a chunk that was actually retrieved.
        A citation to a chunk_id the retriever never returned is a
        phantom citation -- the caller can't verify it, and the model
        is hallucinating a source it never saw."""
        retrieved_ids = {c.chunk_id for c in self.retrieved_chunks}
        for citation in self.citations:
            if citation.chunk_id not in retrieved_ids:
                raise ValueError(
                    f"Citation references chunk_id {citation.chunk_id!r}, "
                    "which is not in retrieved_chunks. Phantom citation: "
                    "the model cited a chunk that was never retrieved. "
                    f"Retrieved: {sorted(retrieved_ids)}"
                )
        return self

    @model_validator(mode="after")
    def _low_confidence_requires_reason(self) -> PolicyAnswer:
        """If the model says confidence=low, it must explain why. An
        unexplained low-confidence answer is worse than no answer --
        the caller has no signal for how to follow up."""
        if self.confidence == "low" and not (self.unanswered_reason or "").strip():
            raise ValueError(
                "confidence='low' requires a non-empty unanswered_reason. "
                "The model must explain why the retrieved passages don't "
                "answer the question rather than silently giving up."
            )
        return self

    @model_validator(mode="after")
    def _high_medium_confidence_requires_citations(self) -> PolicyAnswer:
        """An answer with high or medium confidence but zero citations
        is a hallucination -- the model claims it's supported but shows
        no evidence. Force at least one citation for those cases."""
        if self.confidence in ("high", "medium") and not self.citations:
            raise ValueError(
                f"confidence={self.confidence!r} requires at least one "
                "citation. An answer without evidence is a hallucination; "
                "if the retrieved passages don't support the claim, set "
                "confidence='low' with an unanswered_reason instead."
            )
        return self

    @model_validator(mode="after")
    def _citations_quoted_text_is_substring_of_retrieved(self) -> PolicyAnswer:
        """Every citation's quoted_text must be a substring of the
        chunk it claims (whitespace-normalized). Guards against a model
        that paraphrases quotes instead of pulling verbatim substrings.
        Same [C5] pattern as agents #02/#04/#05/#06/#07/#08."""
        chunks_by_id = {c.chunk_id: c for c in self.retrieved_chunks}
        for citation in self.citations:
            chunk = chunks_by_id.get(citation.chunk_id)
            if chunk is None:
                # The prior validator already handles this; skip.
                continue
            if not _substring_of_normalized(citation.quoted_text, chunk.text):
                raise ValueError(
                    f"Citation quoted_text is not a verbatim substring of "
                    f"chunk {citation.chunk_id!r}. Model paraphrased instead "
                    "of pulling verbatim; caller cannot verify the quote. "
                    f"Quoted: {citation.quoted_text[:80]!r}"
                )
        return self


def _substring_of_normalized(needle: str, haystack: str) -> bool:
    """Whitespace-normalized substring check. Both sides get their
    runs of whitespace collapsed to a single space; case-sensitive
    match. Prevents "verbatim except for a space" being flagged as
    a paraphrase (common when the model reformats markdown)."""

    def _norm(s: str) -> str:
        return " ".join(s.split())

    return _norm(needle) in _norm(haystack)
