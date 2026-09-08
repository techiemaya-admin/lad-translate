#!/usr/bin/env python3
"""
Can this machine transcribe fast enough to run an event?

The pipeline emits every `--emit` seconds over a `--window` second window, so
STT has to finish one window inside one emit interval. Miss that and the
backlog guard starts shedding audio, which the hall hears as gaps rather than
as an error - the failure this repo has already measured at 88.5% on a bad
configuration.

This is the check that replaced "does it have a GPU". It does not, and the
answer to whether that matters is a number rather than an opinion.

    python deploy/vm/benchmark_stt.py --model small

Exits non-zero if p95 misses the budget, so it can gate a provisioning run.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE = ROOT / "fixtures" / "holmes.wav"


def load_16k_mono(path: Path) -> np.ndarray:
    """faster-whisper wants float32 mono at 16kHz; the fixtures are 22050Hz."""
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--window", type=float, default=6.0, help="seconds per transcribe call")
    ap.add_argument("--emit", type=float, default=3.0, help="the budget: one emit interval")
    ap.add_argument("--threads", type=int, default=0, help="0 lets ctranslate2 decide")
    args = ap.parse_args()

    if not FIXTURE.exists():
        print(f"missing fixture {FIXTURE}", file=sys.stderr)
        return 2

    from faster_whisper import WhisperModel

    audio = load_16k_mono(FIXTURE)
    win = int(args.window * 16000)
    if len(audio) < win * 2:
        print("fixture is shorter than two windows", file=sys.stderr)
        return 2

    compute = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute,
                         cpu_threads=args.threads)

    # Warm: the first call pays for model load and lazy kernel setup, and
    # including it would describe a cold start rather than a running session.
    list(model.transcribe(audio[:win], language="en", beam_size=1)[0])

    times = []
    for start in range(0, len(audio) - win, win):
        t0 = time.monotonic()
        list(model.transcribe(audio[start:start + win], language="en", beam_size=1)[0])
        times.append(time.monotonic() - t0)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[min(int(len(times) * 0.95), len(times) - 1)]
    ok = p95 < args.emit

    print(f"\n  model    {args.model} ({args.device}, {compute})")
    print(f"  windows  {len(times)} x {args.window}s")
    print(f"  p50      {p50:.2f}s")
    print(f"  p95      {p95:.2f}s")
    print(f"  budget   {args.emit:.2f}s  (one emit interval)")
    print(f"  headroom {args.emit / p95:.2f}x on p95")
    print(f"\n  {'PASS' if ok else 'FAIL - this machine will shed audio'}\n")

    # p95 rather than p50 on purpose. A median that fits while the tail does not
    # is a session that mostly works and drops words under load, which is the
    # hardest kind of fault to be told about after the fact.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
