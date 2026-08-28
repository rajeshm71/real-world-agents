# Example audio

`sample_episode.wav` is a 3-second, 440 Hz sine tone generated at
author-time by `build_example_audio.py` using only the Python `wave`
stdlib module. No third-party audio content, no external deps, no
provenance issues.

The clip exists so the CLI demo can point at a file that really
exists on disk:

```bash
LLM_PROVIDER=mock uv run python -m agent process examples/sample_episode.wav
```

Real Whisper transcription of a pure tone yields nothing useful. That
is fine -- the transcriber path is tested via monkey-patched stubs,
not against this file. Point the CLI at a real podcast clip you have
locally, or at a YouTube URL, to exercise the full non-mock pipeline:

```bash
uv run python -m agent process ~/podcasts/my_episode.mp3
uv run python -m agent process https://youtube.com/watch?v=...
```
