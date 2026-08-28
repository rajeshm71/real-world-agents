"""Pydantic models for the podcast processor.

Six types travel the three-stage pipeline (fetch -> transcribe ->
structure):

- `AudioSource`:        the fetched audio -- either a local file or
                        the result of a yt-dlp YouTube download.
- `TranscriptSegment`:  one Whisper-emitted time-stamped segment.
- `Transcript`:         the full transcript with duration + joined
                        full_text convenience field.
- `Chapter`:            one chapter of the episode -- title, bounds,
                        one-sentence description.
- `KeyQuote`:           one verbatim quote from the transcript with
                        an approximate timestamp.
- `EpisodeStructure`:   the LLM's output -- title + summary + chapters
                        + key_quotes.
- `PodcastResult`:      full pipeline return, wrapping the source,
                        transcript, structure, and run metadata.
                        Cross-field validators fire here since they
                        need both `transcript` and `structure`.
"""

from __future__ import annotations

import re
from itertools import pairwise
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


class AudioSource(BaseModel):
    """The fetched audio the pipeline transcribes.

    `origin='local'` means the path already existed on disk;
    `origin='youtube'` means the fetcher downloaded it via yt-dlp
    and `url` is the source URL.
    """

    path: Path
    origin: Literal["local", "youtube"]
    url: str | None = None

    @model_validator(mode="after")
    def _url_iff_youtube(self) -> AudioSource:
        if self.origin == "youtube" and not self.url:
            raise ValueError("origin='youtube' requires a source url.")
        if self.origin == "local" and self.url is not None:
            raise ValueError("origin='local' must not carry a url.")
        return self


class TranscriptSegment(BaseModel):
    """One time-stamped segment as faster-whisper emits it."""

    start_seconds: float = Field(..., ge=0)
    end_seconds: float
    text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> TranscriptSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"TranscriptSegment end_seconds ({self.end_seconds}) must be "
                f"strictly greater than start_seconds ({self.start_seconds})."
            )
        return self


class Transcript(BaseModel):
    """The full transcript.

    `full_text` is a convenience field -- the caller derives it from
    segments so downstream code (system prompt building, key-quote
    substring check) doesn't have to re-join. The validator checks
    the two are consistent under whitespace normalization.
    """

    segments: list[TranscriptSegment] = Field(..., min_length=1)
    duration_seconds: float = Field(..., ge=0)
    full_text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _segments_are_monotonic(self) -> Transcript:
        for prev, curr in pairwise(self.segments):
            if curr.start_seconds < prev.start_seconds:
                raise ValueError(
                    f"segments must be non-decreasing in start_seconds; "
                    f"segment at {curr.start_seconds}s follows {prev.start_seconds}s."
                )
        return self

    @model_validator(mode="after")
    def _full_text_matches_segments(self) -> Transcript:
        derived = " ".join(s.text for s in self.segments)
        if _normalize_whitespace(self.full_text) != _normalize_whitespace(derived):
            raise ValueError(
                "full_text must be the whitespace-normalized concatenation "
                "of segment texts; did the caller forget to re-derive it?"
            )
        return self


class Chapter(BaseModel):
    title: str = Field(..., min_length=1)
    start_seconds: float = Field(..., ge=0)
    end_seconds: float
    description: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> Chapter:
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"Chapter end_seconds ({self.end_seconds}) must be "
                f"greater than start_seconds ({self.start_seconds})."
            )
        return self


class KeyQuote(BaseModel):
    quoted_text: str = Field(..., min_length=1)
    approx_start_seconds: float = Field(..., ge=0)


class EpisodeStructure(BaseModel):
    """The LLM's output. Chapters count + key_quotes count are bounded
    here; the transcript-relative validators live on PodcastResult
    since they need the transcript to cross-check against."""

    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    chapters: list[Chapter] = Field(..., min_length=1)
    key_quotes: list[KeyQuote] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def _chapters_non_overlapping_and_monotonic(self) -> EpisodeStructure:
        for prev, curr in pairwise(self.chapters):
            if curr.start_seconds < prev.start_seconds:
                raise ValueError(
                    f"chapters must be sorted by start_seconds; "
                    f"got {curr.start_seconds}s after {prev.start_seconds}s."
                )
            if curr.start_seconds < prev.end_seconds:
                raise ValueError(
                    f"chapters must not overlap; chapter starting at "
                    f"{curr.start_seconds}s begins before previous chapter "
                    f"ended at {prev.end_seconds}s."
                )
        return self


class PodcastResult(BaseModel):
    """Full pipeline return.

    The transcript-relative validators (chapter bounds within
    duration; key-quote is a verbatim substring of full_text) fire
    here since they need both `transcript` and `structure`.
    """

    source: AudioSource
    transcript: Transcript
    structure: EpisodeStructure
    run_meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _chapters_within_duration(self) -> PodcastResult:
        dur = self.transcript.duration_seconds
        for ch in self.structure.chapters:
            if ch.end_seconds > dur + 0.5:
                # +0.5 slack: Whisper's reported duration and chapter
                # end can round differently; be lenient about half a
                # second at the very end of the episode.
                raise ValueError(
                    f"chapter {ch.title!r} ends at {ch.end_seconds}s but "
                    f"transcript duration is {dur}s."
                )
        return self

    @model_validator(mode="after")
    def _key_quotes_are_verbatim(self) -> PodcastResult:
        for kq in self.structure.key_quotes:
            if not _substring_of_normalized(
                kq.quoted_text, self.transcript.full_text
            ):
                raise ValueError(
                    f"key quote {kq.quoted_text[:60]!r} is not a verbatim "
                    "substring of the transcript (whitespace-normalized); "
                    "model must not paraphrase or invent quotes."
                )
        return self
