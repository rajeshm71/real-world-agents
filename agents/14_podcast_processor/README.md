# Podcast episode processor (audio → transcript → chapters + summary)

Give the agent a local podcast episode or a YouTube URL. Get back a title, a two-to-four-sentence summary, ordered non-overlapping chapters with start / end timestamps and one-sentence descriptions, and a short list of verbatim key-quote pull-outs. Transcription runs LOCALLY via faster-whisper (no per-run API cost, no audio ever leaves the box); structuring runs on Anthropic Claude Sonnet against the transcript text. First agent in the catalog that processes the audio modality.

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by the shipped test suite |
| Fetcher (local .wav / .mp3, URL routing, unsupported suffix, missing file) | Fully covered |
| Transcriber (stub-whisper round-trip, empty output guard, monotonic segments) | Fully covered via a duck-typed WhisperModel stub -- no real model download in tests |
| Structurer (JSON fence stripping, retry-once-on-bad-JSON, gives up after retry) | Fully covered |
| Schema validators (audio-source url/local invariant, transcript full_text match, chapter monotonic + non-overlapping, chapter within duration + slack, key-quote verbatim substring) | Fully covered, both directions |
| Real faster-whisper transcription (`LLM_PROVIDER=anthropic`, real model download) | **Not yet verified against a live transcription run.** Structural correctness proven via the stub-whisper tests. Same open-item status as every other agent's first ship. |
| Real Anthropic structuring | Not yet verified against a live API call. |

## Technique demonstrated

**Three-stage audio pipeline with cross-field-verified quotes.** Fetch, transcribe, structure -- each stage is one function in `agent.py`, each stage is independently mock-able / test-able. Same "one file, no framework" shape as #02, #10, #13; the new pieces are:

1. Audio modality (every prior agent processed text or images).
2. Local Whisper via faster-whisper's CTranslate2 backend -- ~1 GB model on first run cached to `~/.cache/huggingface/hub/`, subsequent runs hit disk. No API cost per transcription.
3. YouTube ingestion via yt-dlp (audio-only download to a `tempfile.TemporaryDirectory` that's cleaned up on exit).
4. Cross-field validators on `PodcastResult` that need both the transcript and the LLM-produced structure: chapters must fit within `transcript.duration_seconds`, and every `key_quote.quoted_text` must be a whitespace-normalized verbatim substring of `transcript.full_text`.

## Why this technique for this use case

Podcast processing is a natural three-stage pipeline. Splitting fetch / transcribe / structure into separate functions:

- Lets the reader trace what each stage does and swap any one out independently (local Whisper for OpenAI Whisper API, YouTube for a different source, Anthropic for OpenAI, etc.).
- Keeps the transcript on your machine even when the audio came from YouTube -- only the transcript text is sent to the structuring LLM.
- Makes the mock path trivial: the mock returns a scripted PodcastResult without touching faster-whisper, yt-dlp, or Anthropic. CI never downloads a 1 GB model.

Where this technique is NOT the right fit: episodes shorter than ~2 minutes (chaptering doesn't add value); multi-speaker interviews where speaker diarization matters (Whisper doesn't do it; see "Where this fails"); live-streaming ingestion (v1 is batch: file in, structure out).

## What it does

**`process_episode(source, ...)`** returns a `PodcastResult`:

- `source` -- the fetched `AudioSource` (`origin="local"` or `origin="youtube"` with `url`).
- `transcript` -- `Transcript` with `segments` (time-stamped), `duration_seconds`, `full_text`.
- `structure` -- `EpisodeStructure` with `title`, `summary`, `chapters`, `key_quotes`.
- `run_meta` -- provider, model, whisper_size, duration, per-stage elapsed seconds.

Under `LLM_PROVIDER=mock`, `process_episode` returns a scripted PodcastResult without touching any external dep. The source stem is echoed into the mock title (anti-refactor guard).

## How to run locally

Four commands from a fresh clone:

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set ANTHROPIC_API_KEY (or LLM_PROVIDER=mock)
cd agents/14_podcast_processor
```

Mock demo (no API key, no model download, canned result):

```bash
LLM_PROVIDER=mock uv run python -m agent process examples/sample_episode.wav
```

Real run against a local audio file (first Whisper call downloads ~1 GB into your Hugging Face cache):

```bash
uv run python -m agent process ~/podcasts/episode.mp3 --whisper-size small
```

YouTube URL (audio-only download via yt-dlp):

```bash
uv run python -m agent process https://www.youtube.com/watch?v=... --whisper-size base
```

Rough cost per real run:
- **Whisper transcription**: $0 (local, CPU-workable). First run downloads ~1 GB; subsequent runs are cache-hit. A 60-minute episode transcribes in roughly 5-15 minutes on CPU with `--whisper-size small --compute-type int8`, or under 2 minutes on a modern GPU with `--device cuda`.
- **Anthropic structuring**: **~$0.05 per 60-minute episode** at Claude Sonnet-5 pricing (~12K input tokens transcript + ~1K output tokens structure). Scales linearly with episode length. This is an estimate until a live run pins it.

Whisper size trade-off: `tiny` (75 MB, fast, lower accuracy) → `base` (150 MB) → `small` (500 MB, the default) → `medium` (1.5 GB) → `large-v3` (3 GB, best accuracy). Start with `small`; drop to `base` if CPU is tight.

## Code walkthrough

- `schemas.py`: `AudioSource` (origin invariant), `TranscriptSegment` (end > start), `Transcript` (segments monotonic + whitespace-tolerant `full_text` match), `Chapter`, `KeyQuote`, `EpisodeStructure` (chapters non-overlapping + capped key_quotes), `PodcastResult` (transcript-relative validators: chapters within duration + slack, key-quotes verbatim). `_normalize_whitespace` + `_substring_of_normalized` mirror #10's helpers.
- `agent.py` (`_fetch_audio`, `_ytdlp_download`): stage A. URL detection routes YouTube; unsupported suffix or missing file raises `PodcastError`. `_ytdlp_download` is monkey-patchable in tests.
- `agent.py` (`_transcribe`, `_build_whisper`): stage B. Materializes faster-whisper's segment generator into a validated `Transcript`. Lazy import so mock mode doesn't need the library.
- `agent.py` (`_structure_transcript`, `_parse_and_validate`): stage C. Anthropic Messages call, retry-once-on-validation-failure with the schema error appended to the second-attempt prompt. `_parse_and_validate` strips the JSON code-fence wrappers the model sometimes adds. Same hand-rolled shape as #10.
- `agent.py` (`_translate_api_error`): R5 six-branch translator, matching agents #02-#11 and #13.
- `prompts/system.txt`: rubric for chapter count, boundaries, key quotes; user message includes `Duration:` so the model can pick reasonable bounds.

## When to use / When NOT to use

**Use this pattern when:**
- Audio is your input modality and you need structured output (chapters, summary, quotes).
- You want the transcription step to stay local (privacy, cost, or offline).
- Your episodes fit in one LLM call (~2 hours max at Claude Sonnet's context).

**Do NOT use this pattern when:**
- You need speaker diarization for multi-voice content (see below).
- You need streaming / realtime transcription (batch v1 only).
- Your episodes are so long that the transcript exceeds the LLM context (~3 hours+). Chunk-and-summarize would be needed.

## Where this fails

- **No speaker diarization.** Whisper transcribes but does not label speakers. For interviews / panels, chapters and quotes are still useful but attribution is not. `pyannote.audio` is the standard add-on but drags in torch and needs a Hugging Face account for the pretrained model.
- **1 GB first-run model download.** The first invocation with each `--whisper-size` value blocks on network. Tests never hit this (they use a stub), but a first-time local run will surprise you.
- **YouTube URLs subject to yt-dlp's cat-and-mouse.** YouTube frequently changes its extraction protocol; if yt-dlp fails, the error surfaces through `PodcastError` at `stage="fetch"`. Fix is usually `pip install -U yt-dlp`.
- **Silence-heavy audio produces sparse segments.** `vad_filter=True` on the Whisper call skips long silences, which is usually right, but a podcast with dramatic pauses may have chapters that seem to jump.
- **Pure-tone or non-speech audio** (like the shipped `sample_episode.wav`) produces zero segments in real transcription -- the loader raises `PodcastError`. The sample is there so the CLI mock demo has a file to point at; it is not meant for real transcription.
- **Model can invent chapter titles.** The verbatim check applies to key_quotes only. Chapter titles and descriptions are model-generated free text.
- **Duration slack of 0.5s at the very end.** Chapter end_seconds may exceed reported duration by up to half a second (Whisper rounds differently). A model producing wildly-out-of-bounds chapters (e.g., 100s over) is caught and raised.

## Roadmap (post-v1 improvements)

- OpenAI Whisper API path (`--provider openai-whisper`) as a cloud alternative to local faster-whisper.
- Speaker diarization via pyannote (opt-in, adds torch).
- Arbitrary HTTP(s) audio URLs (Spotify, Apple Podcasts, private feeds).
- Live RSS feed watcher wrapping this agent as a subprocess.
- Chunk-and-summarize path for episodes that exceed the LLM context window.
- Configurable chapter count and title style via a rubric file.
