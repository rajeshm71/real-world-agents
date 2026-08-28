"""Regenerate the tiny synthetic sample_episode.wav.

Stdlib-only (uses `wave` and `math`), no external deps. Not part of
CI; run once when the sample needs to change. The output is committed
as a binary.

    python examples/build_example_audio.py

The clip is a 3-second, 440 Hz sine tone. It's just enough for the
CLI demo (in mock mode) and the loader-round-trip tests to reference
a real file on disk. Real Whisper transcription of a pure tone yields
nothing useful, which is fine -- the transcribe stage is exercised
via monkey-patched stubs, not against this file.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

_HERE = Path(__file__).parent
_OUT = _HERE / "sample_episode.wav"


def main() -> int:
    sample_rate = 16000
    duration_seconds = 3.0
    frequency_hz = 440.0
    n_frames = int(sample_rate * duration_seconds)
    with wave.open(str(_OUT), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_frames):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            wf.writeframesraw(struct.pack("<h", value))
    print(f"wrote {_OUT} ({_OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
