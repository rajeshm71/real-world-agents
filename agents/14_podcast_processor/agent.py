"""Podcast episode processor: agent #14 of real-world-agents.

Technique demonstrated: **audio -> transcript -> structure pipeline
with cross-field-verified quotes.** First agent in the catalog that
processes the audio modality. Three stages, one function each:

1. `_fetch_audio(source)`  -- local file OR YouTube URL (yt-dlp).
2. `_transcribe(source)`   -- faster-whisper local model (~1 GB
                              first-run download; CPU int8 by default).
3. `_structure_transcript(transcript)` -- Anthropic Claude Sonnet:
                              transcript in, JSON out with title,
                              summary, ordered chapters, verbatim key
                              quotes.

Why hand-rolled JSON-validate-retry (no framework): same "here is
what a framework would wrap" pedagogy as #02 and #10. `_parse_and_
validate` strips ```json fences, retries once on validation failure
with the schema error appended to the next prompt.

Provider stance: Anthropic-only in v1 (matches #10; long-context
transcripts are Sonnet's home turf). Multi-provider is a documented
follow-up since only `_call_anthropic` knows which SDK is in play.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

try:
    from .schemas import (
        AudioSource,
        Chapter,
        EpisodeStructure,
        KeyQuote,
        PodcastResult,
        Transcript,
        TranscriptSegment,
    )
except ImportError:
    from schemas import (
        AudioSource,
        Chapter,
        EpisodeStructure,
        KeyQuote,
        PodcastResult,
        Transcript,
        TranscriptSegment,
    )

# --- Constants -------------------------------------------------------------

SUPPORTED_PROVIDERS = ("anthropic",)
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "claude-sonnet-5"
_DEFAULT_WHISPER_SIZE = "small"
_DEFAULT_COMPUTE_TYPE = "int8"
_DEFAULT_DEVICE = "cpu"
_MAX_TOKENS = 4096
_RETRY_LIMIT = 1
_SUPPORTED_AUDIO_SUFFIXES = (".mp3", ".m4a", ".wav", ".flac", ".ogg")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


# --- Error type ------------------------------------------------------------


@dataclass
class ProcessAttempt:
    """Partial state attached to PodcastError. `stage` names where
    things went wrong; `raw_output` and `transcript` are populated
    when relevant."""

    stage: Literal["fetch", "transcribe", "structure", "assemble"]
    url: str | None = None
    raw_output: str = ""
    transcript: Transcript | None = None


class PodcastError(Exception):
    def __init__(self, message: str, partial: ProcessAttempt | None = None):
        super().__init__(message)
        self.message = message
        self.partial = partial


# --- Provider resolution ---------------------------------------------------


def resolve_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    if provider != "mock" and provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Expected 'mock' or one of {SUPPORTED_PROVIDERS}."
        )
    return provider


@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# --- Stage A: fetch --------------------------------------------------------


_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _fetch_audio(source: str | Path, *, tmpdir: Path) -> AudioSource:
    """Route to yt-dlp for URLs; validate local files otherwise."""
    source_str = str(source)
    if _URL_RE.match(source_str):
        try:
            audio_path = _ytdlp_download(source_str, tmpdir)
        except Exception as exc:
            raise PodcastError(
                f"yt-dlp failed on {source_str!r}: "
                f"{type(exc).__name__}: {exc}.",
                partial=ProcessAttempt(stage="fetch", url=source_str),
            ) from exc
        return AudioSource(path=audio_path, origin="youtube", url=source_str)
    path = Path(source)
    if not path.exists() or not path.is_file():
        raise PodcastError(
            f"audio file {source_str!r} does not exist.",
            partial=ProcessAttempt(stage="fetch"),
        )
    if path.suffix.lower() not in _SUPPORTED_AUDIO_SUFFIXES:
        raise PodcastError(
            f"unsupported audio format {path.suffix!r}; supported: "
            f"{list(_SUPPORTED_AUDIO_SUFFIXES)}.",
            partial=ProcessAttempt(stage="fetch"),
        )
    return AudioSource(path=path.resolve(), origin="local", url=None)


def _ytdlp_download(url: str, dest_dir: Path) -> Path:
    """Download best audio via yt-dlp. Isolated so tests can monkey-
    patch it without touching yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise PodcastError(
            "yt-dlp not installed. Run `uv sync --all-packages` from the "
            "repo root."
        ) from exc
    outtmpl = str(dest_dir / "audio.%(ext)s")
    with yt_dlp.YoutubeDL(
        {
            "format": "bestaudio",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
        }
    ) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)
    return Path(downloaded)


# --- Stage B: transcribe ---------------------------------------------------


def _transcribe(
    source: AudioSource,
    *,
    whisper_size: str,
    compute_type: str,
    device: str,
    _whisper_model: Any = None,
) -> Transcript:
    """Run faster-whisper locally. First call downloads the model
    (~1 GB) into ~/.cache/huggingface/hub/; subsequent calls hit disk
    cache."""
    model = _whisper_model if _whisper_model is not None else _build_whisper(
        whisper_size, device=device, compute_type=compute_type
    )
    try:
        segments_iter, info = model.transcribe(
            str(source.path), beam_size=1, vad_filter=True
        )
        raw_segments = list(segments_iter)
    except Exception as exc:
        raise PodcastError(
            f"faster-whisper failed on {source.path.name!r}: "
            f"{type(exc).__name__}: {exc}.",
            partial=ProcessAttempt(stage="transcribe"),
        ) from exc
    try:
        segments = [
            TranscriptSegment(
                start_seconds=float(seg.start),
                end_seconds=float(seg.end),
                text=seg.text,
            )
            for seg in raw_segments
        ]
    except ValidationError as exc:
        # Whisper very occasionally emits a segment with end<=start on
        # very short clips; surface this as a distinct failure so a
        # reader is not misled into thinking faster-whisper itself
        # crashed.
        raise PodcastError(
            f"faster-whisper emitted a segment that failed schema "
            f"validation (e.g. end<=start on very short audio): {exc}",
            partial=ProcessAttempt(stage="transcribe"),
        ) from exc
    if not segments:
        raise PodcastError(
            f"faster-whisper produced zero segments for {source.path.name!r}; "
            "the audio may be silent, contain no detectable speech (VAD "
            "filter removed everything), or be corrupt.",
            partial=ProcessAttempt(stage="transcribe"),
        )
    # Strip per-segment whitespace so a leading space from Whisper
    # doesn't double up with the join separator and inflate token cost.
    full_text = " ".join(s.text.strip() for s in segments)
    return Transcript(
        segments=segments,
        duration_seconds=float(info.duration),
        full_text=full_text,
    )


def _build_whisper(size: str, *, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PodcastError(
            "faster-whisper not installed. Run `uv sync --all-packages` from "
            "the repo root."
        ) from exc
    return WhisperModel(size, device=device, compute_type=compute_type)


# --- Stage C: structure ----------------------------------------------------


def _structure_transcript(
    transcript: Transcript,
    *,
    model: str,
    _anthropic_client: Any = None,
) -> EpisodeStructure:
    """Anthropic Messages call. Retry-once-on-validation-failure with
    the schema error appended so the model can correct itself."""
    client = _anthropic_client if _anthropic_client is not None else _build_anthropic()
    system_prompt = _load_system_prompt()
    user_prompt = (
        f"Duration: {transcript.duration_seconds:.1f}s\n\n"
        f"Transcript:\n{transcript.full_text}"
    )
    last_raw = ""
    last_error = ""
    for attempt in range(_RETRY_LIMIT + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            user_prompt
                            if attempt == 0
                            else (
                                f"{user_prompt}\n\n"
                                f"Previous attempt failed schema validation "
                                f"with:\n{last_error}\n\nReturn JSON that "
                                "passes validation."
                            )
                        ),
                    }
                ],
            )
        except Exception as exc:
            raise _translate_api_error(
                exc,
                partial=ProcessAttempt(
                    stage="structure",
                    raw_output=last_raw,
                    transcript=transcript,
                ),
            ) from exc
        last_raw = _extract_text(response, transcript=transcript)
        try:
            return _parse_and_validate(last_raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt >= _RETRY_LIMIT:
                raise PodcastError(
                    f"structuring failed after {attempt + 1} attempt(s): "
                    f"{exc}",
                    partial=ProcessAttempt(
                        stage="structure",
                        raw_output=last_raw,
                        transcript=transcript,
                    ),
                ) from exc
    raise PodcastError(  # unreachable
        "structuring loop exited without a result.",
        partial=ProcessAttempt(stage="structure", raw_output=last_raw),
    )


def _extract_text(response: Any, *, transcript: Transcript | None = None) -> str:
    """Anthropic returns a list of content blocks; the JSON is in the
    first TextBlock. Empty content is a real error state -- attach a
    partial so the caller can see the transcript context that led to
    it."""
    text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise PodcastError(
            "Anthropic response had no text content blocks (may be a "
            "refusal or a tool-use-only response).",
            partial=ProcessAttempt(stage="structure", transcript=transcript),
        )
    return text_blocks[0]


def _parse_and_validate(raw: str) -> EpisodeStructure:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    data = json.loads(stripped)
    return EpisodeStructure.model_validate(data)


def _build_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise PodcastError(
            "anthropic SDK not installed. Run `uv sync --all-packages` from "
            "the repo root."
        ) from exc
    return anthropic.Anthropic()


# --- Public entry point ----------------------------------------------------


def process_episode(
    source: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    whisper_size: str = _DEFAULT_WHISPER_SIZE,
    compute_type: str = _DEFAULT_COMPUTE_TYPE,
    device: str = _DEFAULT_DEVICE,
    _anthropic_client: Any = None,
    _whisper_model: Any = None,
) -> PodcastResult:
    """Full pipeline: fetch -> transcribe -> structure -> assemble."""
    try:
        resolved = provider or resolve_provider()
    except ValueError as exc:
        raise PodcastError(str(exc)) from exc

    if resolved == "mock":
        return _mock_result(source)

    resolved_model = model or _DEFAULT_MODEL
    stage_times: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="podcast_processor_") as tmp:
        tmp_path = Path(tmp)
        t0 = time.perf_counter()
        audio_source = _fetch_audio(source, tmpdir=tmp_path)
        stage_times["fetch"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        transcript = _transcribe(
            audio_source,
            whisper_size=whisper_size,
            compute_type=compute_type,
            device=device,
            _whisper_model=_whisper_model,
        )
        stage_times["transcribe"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        structure = _structure_transcript(
            transcript,
            model=resolved_model,
            _anthropic_client=_anthropic_client,
        )
        stage_times["structure"] = round(time.perf_counter() - t0, 3)

        try:
            return PodcastResult(
                source=audio_source,
                transcript=transcript,
                structure=structure,
                run_meta={
                    "provider": resolved,
                    "model": resolved_model,
                    "whisper_size": whisper_size,
                    "duration_seconds": transcript.duration_seconds,
                    "stage_seconds": stage_times,
                },
            )
        except ValidationError as exc:
            raise PodcastError(
                f"structure failed transcript-relative validation: {exc}",
                partial=ProcessAttempt(
                    stage="assemble",
                    raw_output=structure.model_dump_json(),
                    transcript=transcript,
                ),
            ) from exc


# --- Error translation (R5) ------------------------------------------------


def _translate_api_error(
    exc: Exception, *, partial: ProcessAttempt | None = None
) -> PodcastError:
    exc_class_name = type(exc).__name__.lower()
    message_lower = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if "ratelimiterror" in exc_class_name:
        return _rate_limit_error(partial)
    if "authenticationerror" in exc_class_name or "apikeyerror" in exc_class_name:
        return _auth_error(partial)
    if status == 429:
        return _rate_limit_error(partial)
    if status == 401:
        return _auth_error(partial)
    if "rate limit" in message_lower or "overloaded" in message_lower:
        return _rate_limit_error(partial)
    if "authentication" in message_lower or "api key" in message_lower:
        return _auth_error(partial)
    return PodcastError(
        f"LLM call failed: {type(exc).__name__}: {exc}.",
        partial=partial,
    )


def _rate_limit_error(partial: ProcessAttempt | None) -> PodcastError:
    return PodcastError(
        "Anthropic is temporarily rate-limited or overloaded. "
        "Wait a minute and try again.",
        partial=partial,
    )


def _auth_error(partial: ProcessAttempt | None) -> PodcastError:
    return PodcastError(
        "Authentication failed: check that ANTHROPIC_API_KEY is set. "
        "See .env.example at the repo root.",
        partial=partial,
    )


# --- Mock mode -------------------------------------------------------------


def _mock_result(source: str | Path) -> PodcastResult:
    """Scripted PodcastResult without touching yt-dlp / faster-whisper /
    anthropic. Title echoes the source stem as the anti-refactor guard."""
    source_path = Path(str(source))
    if _URL_RE.match(str(source)):
        stem = "url_episode"
    else:
        stem = source_path.stem or "episode"

    segments = [
        TranscriptSegment(
            start_seconds=0.0, end_seconds=10.0,
            text="Welcome to the show; today we're covering three topics.",
        ),
        TranscriptSegment(
            start_seconds=10.0, end_seconds=20.0,
            text="First up, we look at what people really want from an agent.",
        ),
        TranscriptSegment(
            start_seconds=20.0, end_seconds=30.0,
            text="Second, we dig into the trade-offs no one is talking about.",
        ),
    ]
    full_text = " ".join(s.text for s in segments)
    transcript = Transcript(
        segments=segments, duration_seconds=30.0, full_text=full_text
    )

    structure = EpisodeStructure(
        title=f"[MOCK] {stem}: three topics episode",
        summary=(
            f"[MOCK summary for source stem of length {len(stem)}] A short "
            "monologue introducing three podcast topics in half a minute."
        ),
        chapters=[
            Chapter(
                title="Intro and setup",
                start_seconds=0.0,
                end_seconds=10.0,
                description="Host greets listeners and previews the episode.",
            ),
            Chapter(
                title="Trade-offs no one talks about",
                start_seconds=10.0,
                end_seconds=30.0,
                description="Two topics unpacked back-to-back.",
            ),
        ],
        key_quotes=[
            KeyQuote(
                quoted_text="the trade-offs no one is talking about",
                approx_start_seconds=20.0,
            )
        ],
    )

    is_url = _URL_RE.match(str(source)) is not None
    return PodcastResult(
        source=AudioSource(
            path=(source_path.resolve() if source_path.exists() else source_path),
            origin="youtube" if is_url else "local",
            url=str(source) if is_url else None,
        ),
        transcript=transcript,
        structure=structure,
        run_meta={
            "provider": "mock",
            "model": "mock",
            "whisper_size": "mock",
            "duration_seconds": 30.0,
            "stage_seconds": {"fetch": 0.0, "transcribe": 0.0, "structure": 0.0},
        },
    )


# --- CLI ------------------------------------------------------------------


def _fmt_timestamp(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="podcast-processor",
        description=(
            "Transcribe a local audio file or a YouTube URL and structure it "
            "into chapters + summary + key quotes. Set LLM_PROVIDER=mock for "
            "a scripted demo, or supply ANTHROPIC_API_KEY for real answers."
        ),
    )
    parser.add_argument("source", type=str, help="Local audio path OR https:// URL.")
    parser.add_argument("--whisper-size", type=str, default=_DEFAULT_WHISPER_SIZE)
    parser.add_argument("--compute-type", type=str, default=_DEFAULT_COMPUTE_TYPE)
    parser.add_argument("--device", type=str, default=_DEFAULT_DEVICE)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--provider", choices=(*SUPPORTED_PROVIDERS, "mock"), default=None
    )
    args = parser.parse_args()

    try:
        result = process_episode(
            args.source,
            provider=args.provider,
            model=args.model,
            whisper_size=args.whisper_size,
            compute_type=args.compute_type,
            device=args.device,
        )
    except PodcastError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.partial is not None and exc.partial.raw_output:
            print(f"raw model output:\n{exc.partial.raw_output}", file=sys.stderr)
        return 1

    out_path = Path(__file__).parent / "last_run.json"
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    s = result.structure
    print(f"Title:    {s.title}")
    print(f"Duration: {_fmt_timestamp(result.transcript.duration_seconds)}")
    print(f"Source:   {result.source.origin} :: {result.source.path.name}")
    print()
    print("Summary:")
    print(s.summary)
    print()
    print(f"Chapters ({len(s.chapters)}):")
    for ch in s.chapters:
        print(
            f"  [{_fmt_timestamp(ch.start_seconds)}-"
            f"{_fmt_timestamp(ch.end_seconds)}] {ch.title}"
        )
        print(f"       {ch.description}")
    if s.key_quotes:
        print()
        print(f"Key quotes ({len(s.key_quotes)}):")
        for kq in s.key_quotes:
            print(f"  [{_fmt_timestamp(kq.approx_start_seconds)}] {kq.quoted_text!r}")
    print()
    print(f"Full trace written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
