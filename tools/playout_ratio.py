#!/usr/bin/env python3
"""
How much audio does each language produce for the same English?

The number that fills the playout queue. A target that runs longer than the
source falls behind by the difference on every phrase, and session/drift.py
has to speed it up or skip. Arabic's policy came from this measurement, made
by hand; German shipped without one and the question came back the first time
a listener said it sounded late.

Read s/100ch, not "vs source". "vs source" divides by how fast the human on
the fixture happened to read: holmes.wav is 839 characters over 75 seconds and
keynote.wav is 446 over 25, so the same voice scores 0.66x on one and 1.05x on
the other. Per 100 characters of the TARGET text is stable across both.

    python tools/playout_ratio.py keynote
    python tools/playout_ratio.py holmes

Needs the voices and MT models, so run it on the VM:

    sudo -u ladtranslate /opt/lad-translate/.venv/bin/python tools/playout_ratio.py

Measured 9 Sep 2026 on the develop VM, both fixtures:

    fr  fr_FR-siwis-medium       5.13 - 5.30 s/100ch
    de  de_DE-thorsten-medium    5.15 - 5.42 s/100ch
    ar  ar_JO-kareem-medium     12.10 - 13.15 s/100ch
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.base import VoiceSpec
from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter
from lad_translate.obs.log import configure

TARGETS = ["fr", "de", "ar"]


async def main() -> int:
    configure("ERROR")

    stem = sys.argv[1] if len(sys.argv) > 1 else "keynote"
    src_text = (ROOT / "fixtures" / f"{stem}.txt").read_text().strip()
    with wave.open(str(ROOT / "fixtures" / f"{stem}.wav")) as w:
        source_s = w.getnframes() / w.getframerate()

    # Sentence at a time, which is how the pipeline actually feeds TTS.
    sentences = [s.strip() + "." for s in src_text.split(".") if s.strip()]

    mt = RoutingMtAdapter(
        "en", TARGETS,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    tts = PiperTtsAdapter(TARGETS, voice_root=ROOT / "models" / "tts")

    print(f"source   {stem}.wav  {source_s:.2f}s of English, {len(src_text)} chars, "
          f"{len(sentences)} sentences\n")

    async with tts:
        rows = []
        for code in TARGETS:
            chars = 0
            audio_s = 0.0
            synth_s = 0.0
            for sentence in sentences:
                out = await mt.translate_many(sentence, "en", [code])
                text = out.get(code, "")
                if not text.strip():
                    continue
                chars += len(text)
                started = time.monotonic()
                spec = VoiceSpec(language=code, voice_id=DEFAULT_VOICES[code], speed=1.0)
                async for chunk in tts.synthesise(text, spec, chunk_id=0):
                    audio_s += len(chunk.pcm) / 2 / chunk.sample_rate
                synth_s += time.monotonic() - started
            rows.append((code, chars, audio_s, synth_s))

    print(f"  {'lang':5s} {'voice':24s} {'audio':>8s} {'vs source':>10s} "
          f"{'s/100ch':>8s} {'synth':>8s} {'RTF':>6s}")
    for code, chars, audio_s, synth_s in rows:
        ratio = audio_s / source_s
        per100 = audio_s / chars * 100 if chars else 0.0
        rtf = synth_s / audio_s if audio_s else 0.0
        print(f"  {code:5s} {DEFAULT_VOICES[code]:24s} {audio_s:7.2f}s "
              f"{ratio:9.2f}x {per100:7.2f}s {synth_s:7.2f}s {rtf:5.2f}")

    print("\n  vs source > 1.0 means the queue grows on every phrase.")
    print("  A whole talk multiplies it: at 1.25x, forty minutes ends ten behind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
