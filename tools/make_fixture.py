#!/usr/bin/env python3
"""
Build a synthetic speech fixture for offline pipeline testing.

Synthesised speech is clean, evenly paced and free of the accents, crosstalk
and room noise that make real conference audio hard. It is enough to prove the
pipeline moves audio end to end. It is not enough to judge transcript accuracy,
and a chunker tuned only on this will be tuned for the easy case.

Replace it with a real venue recording before trusting any number from it.

Usage:
    python tools/make_fixture.py --out fixtures/keynote.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_TEXT = (
    "Good morning everyone, and thank you for joining us today. "
    "Over the next forty minutes I want to walk through three things. "
    "First, where the market actually stands right now. "
    "Second, what we learned from the pilot programme in Sharjah. "
    "And third, what that means for the roadmap. "
    "Revenue across the sector grew eleven percent last year, "
    "but that number hides an enormous amount of variation. "
    "The top four operators took almost all of that growth."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("fixtures/keynote.wav"))
    ap.add_argument("--voice", default="models/tts/en_GB-alba-medium.onnx")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(args.voice)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(args.out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(args.text, wav, set_wav_format=False)

    with wave.open(str(args.out)) as check:
        frames = check.getnframes()
    seconds = frames / voice.config.sample_rate
    print(f"{args.out}  {seconds:.1f}s @ {voice.config.sample_rate}Hz")
    (args.out.with_suffix(".txt")).write_text(args.text + "\n")
    print(f"{args.out.with_suffix('.txt')}  ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
