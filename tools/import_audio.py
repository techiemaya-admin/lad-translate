#!/usr/bin/env python3
"""
Convert any audio file into a fixture WAV.

Decodes with PyAV, which arrives as a faster-whisper dependency, so no ffmpeg
binary is needed. Handles ogg, mp3, m4a, flac and anything else PyAV reads.

Real recordings matter here. Synthesised speech is evenly paced, unaccented and
free of room tone, so a pipeline tuned only on it is tuned for a case that will
never occur at a venue. Import a real clip and the transcript quality drops
immediately, which is the useful signal.

Usage:
    python tools/import_audio.py --in speech.ogg --out fixtures/jfk.wav
    python tools/import_audio.py --in speech.ogg --out fixtures/jfk.wav --start 30 --duration 45
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

TARGET_RATE = 22_050
"""Matches the Piper output rate, so fixtures and output share a clock."""


def convert(source: Path, dest: Path, start: float, duration: float | None, rate: int) -> float:
    import av

    container = av.open(str(source))
    stream = container.streams.audio[0]

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    if start > 0:
        # PyAV seeks in stream time base; nudge back slightly so the first
        # frame after the seek is not clipped mid-packet.
        container.seek(int(max(0.0, start - 0.5) / float(stream.time_base)), stream=stream)

    chunks: list[bytes] = []
    written = 0.0
    for frame in container.decode(audio=0):
        stamp = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
        if stamp < start:
            continue
        for resampled in resampler.resample(frame):
            pcm = bytes(resampled.planes[0])[: resampled.samples * 2]
            chunks.append(pcm)
            written += resampled.samples / rate
        if duration is not None and written >= duration:
            break
    container.close()

    if not chunks:
        raise SystemExit(f"error: no audio decoded from {source} at start={start}s")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(chunks))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="source", type=Path, required=True)
    ap.add_argument("--out", dest="dest", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    ap.add_argument("--duration", type=float, default=None, help="seconds to take")
    ap.add_argument("--rate", type=int, default=TARGET_RATE)
    args = ap.parse_args()

    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 1
    seconds = convert(args.source, args.dest, args.start, args.duration, args.rate)
    print(f"{args.dest}  {seconds:.1f}s @ {args.rate}Hz mono")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
