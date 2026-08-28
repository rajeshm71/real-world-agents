"""Smoke tests for the podcast processor.

Zero real API calls, zero real Whisper runs, zero real network. Every
non-mock path is exercised via monkey-patched `_ytdlp_download` /
`_whisper_model` / `_anthropic_client` injections.

Sections:
1. Mock path (5): full round-trip; stem echoed in title; scripted
   3 segments; 2 chapters; 1 key quote.
2. Fetcher (5).
3. Transcriber (6).
4. Schema validators (10, both directions).
5. Structurer (5): fence stripping + retry.
6. R5 branches (6).
7. Constants + example sanity (3).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_AGENT_DIR.parent))

_agent = importlib.import_module("14_podcast_processor.agent")
_schemas = importlib.import_module("14_podcast_processor.schemas")

process_episode = _agent.process_episode
PodcastError = _agent.PodcastError
ProcessAttempt = _agent.ProcessAttempt
_fetch_audio = _agent._fetch_audio
_structure_transcript = _agent._structure_transcript
_parse_and_validate = _agent._parse_and_validate
_translate_api_error = _agent._translate_api_error
_transcribe = _agent._transcribe
SUPPORTED_PROVIDERS = _agent.SUPPORTED_PROVIDERS
_SUPPORTED_AUDIO_SUFFIXES = _agent._SUPPORTED_AUDIO_SUFFIXES

AudioSource = _schemas.AudioSource
TranscriptSegment = _schemas.TranscriptSegment
Transcript = _schemas.Transcript
Chapter = _schemas.Chapter
KeyQuote = _schemas.KeyQuote
EpisodeStructure = _schemas.EpisodeStructure
PodcastResult = _schemas.PodcastResult

_SAMPLE = _AGENT_DIR / "examples" / "sample_episode.wav"


# --- Helpers ---------------------------------------------------------------


class _StubWhisper:
    """Duck-typed WhisperModel that returns scripted segments."""

    def __init__(self, segments, duration):
        self._segments = segments
        self._duration = duration

    def transcribe(self, path, **kwargs):
        info = types.SimpleNamespace(duration=self._duration)
        segs = [
            types.SimpleNamespace(start=s["start"], end=s["end"], text=s["text"])
            for s in self._segments
        ]
        return iter(segs), info


class _StubAnthropic:
    """Duck-typed client returning scripted text blocks per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = self
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        text = self._responses.pop(0)
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])


def _good_structure_json(duration: float = 30.0) -> str:
    return json.dumps(
        {
            "title": "A good title here",
            "summary": "A short summary. Two sentences.",
            "chapters": [
                {"title": "Intro", "start_seconds": 0.0, "end_seconds": 10.0,
                 "description": "Opening."},
                {"title": "Middle", "start_seconds": 10.0, "end_seconds": 20.0,
                 "description": "Middle content."},
                {"title": "End", "start_seconds": 20.0, "end_seconds": duration,
                 "description": "Wrap up."},
            ],
            "key_quotes": [
                {"quoted_text": "we're covering three topics",
                 "approx_start_seconds": 5.0},
            ],
        }
    )


def _sample_transcript() -> Transcript:
    segs = [
        TranscriptSegment(start_seconds=0.0, end_seconds=10.0,
                          text="Welcome to the show; today we're covering three topics."),
        TranscriptSegment(start_seconds=10.0, end_seconds=20.0,
                          text="First topic content."),
        TranscriptSegment(start_seconds=20.0, end_seconds=30.0,
                          text="Second topic content."),
    ]
    return Transcript(
        segments=segs,
        duration_seconds=30.0,
        full_text=" ".join(s.text for s in segs),
    )


# --- 1. Mock path ----------------------------------------------------------


def test_mock_returns_valid_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = process_episode(_SAMPLE)
    assert isinstance(r, PodcastResult)
    assert r.run_meta["provider"] == "mock"


def test_mock_echoes_source_stem_in_title(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = process_episode(_SAMPLE)
    assert "sample_episode" in r.structure.title


def test_mock_has_three_scripted_segments(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = process_episode(_SAMPLE)
    assert len(r.transcript.segments) == 3


def test_mock_has_two_scripted_chapters(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = process_episode(_SAMPLE)
    assert len(r.structure.chapters) == 2


def test_mock_does_not_import_faster_whisper(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    r = process_episode(_SAMPLE)
    assert r.transcript.duration_seconds == 30.0


# --- 2. Fetcher -----------------------------------------------------------


def test_local_wav_returns_local_audio_source(tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"RIFF")
    src = _fetch_audio(tmp_path / "clip.wav", tmpdir=tmp_path)
    assert src.origin == "local"
    assert src.url is None


def test_local_unsupported_suffix_rejected(tmp_path):
    (tmp_path / "clip.txt").write_text("x", encoding="utf-8")
    with pytest.raises(PodcastError, match="unsupported audio format"):
        _fetch_audio(tmp_path / "clip.txt", tmpdir=tmp_path)


def test_local_missing_file_rejected(tmp_path):
    with pytest.raises(PodcastError, match="does not exist"):
        _fetch_audio(tmp_path / "nope.mp3", tmpdir=tmp_path)


def test_youtube_url_routes_through_ytdlp(monkeypatch, tmp_path):
    calls: list[tuple[str, Path]] = []

    def fake_download(url: str, dest_dir: Path) -> Path:
        calls.append((url, dest_dir))
        out = dest_dir / "audio.m4a"
        out.write_bytes(b"fake")
        return out

    monkeypatch.setattr(_agent, "_ytdlp_download", fake_download)
    src = _fetch_audio("https://youtube.com/watch?v=abc", tmpdir=tmp_path)
    assert src.origin == "youtube"
    assert src.url == "https://youtube.com/watch?v=abc"
    assert calls == [("https://youtube.com/watch?v=abc", tmp_path)]


def test_url_case_insensitive(monkeypatch, tmp_path):
    def fake(url, dest_dir):
        p = dest_dir / "audio.m4a"
        p.write_bytes(b"fake")
        return p
    monkeypatch.setattr(_agent, "_ytdlp_download", fake)
    src = _fetch_audio("HTTPS://youtu.be/x", tmpdir=tmp_path)
    assert src.origin == "youtube"


# --- 3. Transcriber -------------------------------------------------------


def test_transcribe_wraps_stub_whisper_output(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    src = AudioSource(path=audio, origin="local")
    stub = _StubWhisper(
        segments=[
            {"start": 0.0, "end": 5.0, "text": "hello"},
            {"start": 5.0, "end": 10.0, "text": "world"},
        ],
        duration=10.0,
    )
    t = _transcribe(src, whisper_size="small", compute_type="int8",
                    device="cpu", _whisper_model=stub)
    assert t.duration_seconds == 10.0
    assert len(t.segments) == 2
    assert "hello" in t.full_text


def test_transcribe_zero_segments_raises(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    src = AudioSource(path=audio, origin="local")
    stub = _StubWhisper(segments=[], duration=0.0)
    with pytest.raises(PodcastError, match="zero segments"):
        _transcribe(src, whisper_size="small", compute_type="int8",
                    device="cpu", _whisper_model=stub)


def test_transcript_rejects_non_monotonic_segments():
    with pytest.raises(ValidationError, match="non-decreasing"):
        Transcript(
            segments=[
                TranscriptSegment(start_seconds=10.0, end_seconds=15.0, text="b"),
                TranscriptSegment(start_seconds=0.0, end_seconds=5.0, text="a"),
            ],
            duration_seconds=15.0,
            full_text="b a",
        )


def test_segment_rejects_end_before_start():
    with pytest.raises(ValidationError, match="strictly greater"):
        TranscriptSegment(start_seconds=10.0, end_seconds=5.0, text="x")


def test_transcript_full_text_must_match_segments():
    with pytest.raises(ValidationError, match="whitespace-normalized"):
        Transcript(
            segments=[
                TranscriptSegment(start_seconds=0.0, end_seconds=5.0, text="hello"),
            ],
            duration_seconds=5.0,
            full_text="totally different",
        )


def test_transcript_full_text_whitespace_tolerant():
    """Extra spaces in full_text vs segment joins should still pass."""
    Transcript(
        segments=[
            TranscriptSegment(start_seconds=0.0, end_seconds=5.0, text="hello world"),
        ],
        duration_seconds=5.0,
        full_text="  hello   world  ",
    )


# --- 4. Schema validators (structure-level) -------------------------------


def _good_chapters(duration=30.0) -> list[Chapter]:
    return [
        Chapter(title="A", start_seconds=0.0, end_seconds=10.0, description="a."),
        Chapter(title="B", start_seconds=10.0, end_seconds=20.0, description="b."),
        Chapter(title="C", start_seconds=20.0, end_seconds=duration, description="c."),
    ]


def test_episode_structure_requires_at_least_one_chapter():
    with pytest.raises(ValidationError):
        EpisodeStructure(
            title="t", summary="s", chapters=[], key_quotes=[]
        )


def test_chapters_non_overlapping_positive():
    EpisodeStructure(
        title="t", summary="s", chapters=_good_chapters(), key_quotes=[]
    )


def test_chapters_overlapping_negative():
    with pytest.raises(ValidationError, match="not overlap"):
        EpisodeStructure(
            title="t", summary="s",
            chapters=[
                Chapter(title="A", start_seconds=0.0, end_seconds=15.0, description="a."),
                Chapter(title="B", start_seconds=10.0, end_seconds=20.0, description="b."),
            ],
            key_quotes=[],
        )


def test_chapters_non_monotonic_negative():
    with pytest.raises(ValidationError, match="sorted by start_seconds"):
        EpisodeStructure(
            title="t", summary="s",
            chapters=[
                Chapter(title="A", start_seconds=20.0, end_seconds=25.0, description="a."),
                Chapter(title="B", start_seconds=0.0, end_seconds=10.0, description="b."),
            ],
            key_quotes=[],
        )


def test_key_quotes_capped_at_six():
    with pytest.raises(ValidationError):
        EpisodeStructure(
            title="t", summary="s", chapters=_good_chapters(),
            key_quotes=[
                KeyQuote(quoted_text=f"q{i}", approx_start_seconds=float(i))
                for i in range(7)
            ],
        )


def test_podcast_result_key_quote_verbatim_positive():
    tr = _sample_transcript()
    struct = EpisodeStructure(
        title="t", summary="s", chapters=_good_chapters(),
        key_quotes=[
            KeyQuote(quoted_text="covering three topics", approx_start_seconds=5.0)
        ],
    )
    PodcastResult(
        source=AudioSource(path=Path("/tmp/x.wav"), origin="local"),
        transcript=tr, structure=struct, run_meta={},
    )


def test_podcast_result_key_quote_invented_negative():
    tr = _sample_transcript()
    struct = EpisodeStructure(
        title="t", summary="s", chapters=_good_chapters(),
        key_quotes=[
            KeyQuote(quoted_text="quantum entanglement is amazing",
                     approx_start_seconds=5.0)
        ],
    )
    with pytest.raises(ValidationError, match="verbatim substring"):
        PodcastResult(
            source=AudioSource(path=Path("/tmp/x.wav"), origin="local"),
            transcript=tr, structure=struct, run_meta={},
        )


def test_podcast_result_chapter_beyond_duration_negative():
    tr = _sample_transcript()  # duration = 30
    struct = EpisodeStructure(
        title="t", summary="s",
        chapters=[
            Chapter(title="way past end", start_seconds=25.0,
                    end_seconds=100.0, description="d."),
        ],
        key_quotes=[],
    )
    with pytest.raises(ValidationError, match="transcript duration"):
        PodcastResult(
            source=AudioSource(path=Path("/tmp/x.wav"), origin="local"),
            transcript=tr, structure=struct, run_meta={},
        )


def test_podcast_result_chapter_within_slack_ok():
    """0.5s slack at the very end for rounding differences."""
    tr = _sample_transcript()  # duration = 30
    struct = EpisodeStructure(
        title="t", summary="s",
        chapters=[
            Chapter(title="Barely over", start_seconds=25.0, end_seconds=30.3,
                    description="d."),
        ],
        key_quotes=[],
    )
    PodcastResult(
        source=AudioSource(path=Path("/tmp/x.wav"), origin="local"),
        transcript=tr, structure=struct, run_meta={},
    )


def test_audio_source_youtube_requires_url():
    with pytest.raises(ValidationError, match="requires a source url"):
        AudioSource(path=Path("/tmp/x.wav"), origin="youtube")


# --- 5. Structurer --------------------------------------------------------


def test_parse_and_validate_strips_json_fences():
    raw = "```json\n" + _good_structure_json() + "\n```"
    result = _parse_and_validate(raw)
    assert result.title == "A good title here"


def test_structurer_succeeds_first_try():
    stub = _StubAnthropic(responses=[_good_structure_json()])
    r = _structure_transcript(_sample_transcript(), model="m",
                              _anthropic_client=stub)
    assert stub.call_count == 1
    assert r.title == "A good title here"


def test_structurer_retries_once_on_bad_json():
    stub = _StubAnthropic(responses=["not json at all", _good_structure_json()])
    r = _structure_transcript(_sample_transcript(), model="m",
                              _anthropic_client=stub)
    assert stub.call_count == 2
    assert r.title == "A good title here"


def test_structurer_gives_up_after_retry():
    stub = _StubAnthropic(responses=["not json", "still not json"])
    with pytest.raises(PodcastError, match="structuring failed"):
        _structure_transcript(_sample_transcript(), model="m",
                              _anthropic_client=stub)
    assert stub.call_count == 2


def test_structurer_raw_output_attached_on_failure():
    stub = _StubAnthropic(responses=["nope 1", "nope 2"])
    try:
        _structure_transcript(_sample_transcript(), model="m",
                              _anthropic_client=stub)
    except PodcastError as exc:
        assert exc.partial is not None
        assert "nope 2" in exc.partial.raw_output
    else:
        pytest.fail("expected PodcastError")


# --- 6. R5 error translator -----------------------------------------------


def test_translator_class_rate_limit():
    class RateLimitError(Exception):
        pass
    assert "rate-limited" in _translate_api_error(RateLimitError("x")).message


def test_translator_class_auth():
    class AuthenticationError(Exception):
        pass
    assert "ANTHROPIC_API_KEY" in _translate_api_error(AuthenticationError("x")).message


def test_translator_status_429():
    exc = RuntimeError("x")
    exc.status_code = 429
    assert "rate-limited" in _translate_api_error(exc).message


def test_translator_status_401():
    exc = RuntimeError("x")
    exc.status_code = 401
    assert "ANTHROPIC_API_KEY" in _translate_api_error(exc).message


def test_translator_message_rate_limit():
    assert "rate-limited" in _translate_api_error(
        RuntimeError("Rate limit exceeded")
    ).message


def test_translator_message_auth():
    assert "ANTHROPIC_API_KEY" in _translate_api_error(
        RuntimeError("Invalid API key")
    ).message


def test_translator_generic_fallthrough():
    assert "LLM call failed" in _translate_api_error(
        RuntimeError("some other error")
    ).message


# --- 7. Constants + example sanity ----------------------------------------


def test_supported_providers_anthropic_only():
    assert SUPPORTED_PROVIDERS == ("anthropic",)


def test_supported_audio_suffixes():
    assert ".mp3" in _SUPPORTED_AUDIO_SUFFIXES
    assert ".wav" in _SUPPORTED_AUDIO_SUFFIXES


def test_sample_episode_exists():
    assert _SAMPLE.exists()
    assert _SAMPLE.stat().st_size > 1000


# --- 8. Post-review hardening (H1, M1, M2, L1, L2) ------------------------


def test_extract_text_no_text_blocks_attaches_partial():
    """H1: an Anthropic response with no text blocks (refusal, tool-use
    only, etc.) must raise PodcastError with a partial carrying the
    transcript, not a bare error message."""
    empty_response = types.SimpleNamespace(content=[])
    tr = _sample_transcript()
    with pytest.raises(PodcastError) as exc_info:
        _agent._extract_text(empty_response, transcript=tr)
    assert exc_info.value.partial is not None
    assert exc_info.value.partial.stage == "structure"
    assert exc_info.value.partial.transcript is tr


def test_transcribe_narrow_catch_reports_segment_validation_distinctly(tmp_path):
    """M1: a Whisper segment with end<=start must surface a distinct
    error, not be mislabeled as 'faster-whisper failed'."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    src = AudioSource(path=audio, origin="local")
    # Stub emits a bad segment: end == start.
    stub = _StubWhisper(
        segments=[{"start": 0.0, "end": 0.0, "text": "oops"}],
        duration=1.0,
    )
    with pytest.raises(PodcastError, match="failed schema validation"):
        _transcribe(src, whisper_size="small", compute_type="int8",
                    device="cpu", _whisper_model=stub)


def test_transcribe_strips_leading_spaces_from_segments(tmp_path):
    """M2: Whisper segments often start with a leading space; the
    joined full_text should not have double-space runs."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    src = AudioSource(path=audio, origin="local")
    stub = _StubWhisper(
        segments=[
            {"start": 0.0, "end": 5.0, "text": " hello"},
            {"start": 5.0, "end": 10.0, "text": " world"},
        ],
        duration=10.0,
    )
    t = _transcribe(src, whisper_size="small", compute_type="int8",
                    device="cpu", _whisper_model=stub)
    assert "  " not in t.full_text


def test_zero_segments_error_mentions_vad(tmp_path):
    """L1: the zero-segments message must call out the VAD-filter case
    since our shipped sample_episode.wav triggers exactly that path."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    src = AudioSource(path=audio, origin="local")
    stub = _StubWhisper(segments=[], duration=3.0)
    with pytest.raises(PodcastError, match="VAD filter"):
        _transcribe(src, whisper_size="small", compute_type="int8",
                    device="cpu", _whisper_model=stub)


def test_mock_result_with_url_source_uses_friendly_stem(monkeypatch):
    """L2: passing a URL to the mock (unusual but legal) should
    produce a friendly title and mark the AudioSource as youtube, not
    an ugly `watch?v=xyz` stem stuffed into a `local` source."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    r = process_episode("https://youtube.com/watch?v=xyz")
    assert "watch?v" not in r.structure.title
    assert r.source.origin == "youtube"
    assert r.source.url == "https://youtube.com/watch?v=xyz"
