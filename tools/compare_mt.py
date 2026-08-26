#!/usr/bin/env python3
"""
Compare translation backends on the same audio, ignoring latency.

Transcribes ONCE, then runs the identical chunks through every backend. That
matters: transcribing separately per backend would let STT variance leak into
what is supposed to be a translation comparison, and Whisper does not produce
the same transcript twice on the same file when chunk boundaries move.

Latency is deliberately not measured. Use a large STT model and let each
backend take as long as it likes; the question here is only what the audience
would hear. For latency use tools/session_live.py.

Writes one WAV per backend per language so the difference can be heard, not
just read.

Usage:
    python tools/compare_mt.py --audio fixtures/jfk.wav --targets te,hi,fr
    python tools/compare_mt.py --audio fixtures/jfk.wav --targets te --stt small
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.base import VoiceSpec  # noqa: E402
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter  # noqa: E402
from lad_translate.obs.log import configure  # noqa: E402

MODEL_ROOT = ROOT / "models" / "mt"


def transcribe(audio: Path, model_size: str) -> list[str]:
    """
    One transcript, split into phrase-sized pieces at sentence boundaries.

    Uses batch transcription rather than the streaming adapter: latency is not
    the question here, and batch gives the best transcript the model can
    produce, which is the fairest input for both backends.
    """
    from faster_whisper import WhisperModel

    started = time.monotonic()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio), language="en", beam_size=1)
    pieces = [s.text.strip() for s in segments if s.text.strip()]
    print(f"transcribed with {model_size} in {time.monotonic() - started:.0f}s, "
          f"{len(pieces)} segments\n")
    return pieces


async def build_backends(targets: list[str], want_nllb: bool) -> dict:
    from lad_translate.adapters.mt_opus import OpusMtAdapter

    backends: dict[str, object] = {}
    try:
        backends["opus-mt"] = OpusMtAdapter("en", targets, model_root=MODEL_ROOT)
    except FileNotFoundError as exc:
        print(f"opus-mt unavailable: {exc}\n")

    if want_nllb:
        from lad_translate.adapters.mt_nllb import NllbMtAdapter

        try:
            backends["nllb-200"] = NllbMtAdapter(
                "en", targets, model_path=MODEL_ROOT / "nllb-600m"
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"nllb-200 unavailable: {exc}\n")
    return backends


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "jfk.wav")
    ap.add_argument("--targets", default="te,hi,fr")
    ap.add_argument("--stt", default="small", help="latency is irrelevant here, so use a good one")
    ap.add_argument("--out", type=Path, default=ROOT / "fixtures" / "compare")
    ap.add_argument("--no-audio", action="store_true", help="text comparison only")
    ap.add_argument("--no-nllb", action="store_true")
    args = ap.parse_args()

    configure("ERROR")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    chunks = transcribe(args.audio, args.stt)
    backends = await build_backends(targets, not args.no_nllb)
    if not backends:
        print("no backends available", file=sys.stderr)
        return 1

    # backend -> language -> list of translated strings
    results: dict[str, dict[str, list[str]]] = {
        name: {lang: [] for lang in targets} for name in backends
    }

    for i, chunk in enumerate(chunks):
        print(f"[{i}] EN  {chunk}")
        for name, backend in backends.items():
            started = time.monotonic()
            out = await backend.translate_many(chunk, "en", targets)
            ms = (time.monotonic() - started) * 1000
            for lang in targets:
                results[name][lang].append(out.get(lang, ""))
            label = f"{name} ({ms:.0f}ms)"
            for lang in targets:
                print(f"     {lang}  {out.get(lang, '')}   [{label}]" if lang == targets[0]
                      else f"     {lang}  {out.get(lang, '')}")
        print()

    if args.no_audio:
        return 0

    print("synthesising...")
    args.out.mkdir(parents=True, exist_ok=True)
    voiceable = [t for t in targets if t in DEFAULT_VOICES]
    tts = PiperTtsAdapter(voiceable, voice_root=ROOT / "models" / "tts")
    async with tts:
        for name in backends:
            for lang in voiceable:
                pcm: list[bytes] = []
                for idx, text in enumerate(results[name][lang]):
                    if not text.strip():
                        continue
                    voice = VoiceSpec(language=lang, voice_id=DEFAULT_VOICES[lang])
                    async for speech in tts.synthesise(text, voice, idx):
                        pcm.append(speech.pcm)
                if not pcm:
                    continue
                path = args.out / f"{name}-{lang}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(tts.sample_rate)
                    w.writeframes(b"".join(pcm))
                seconds = sum(len(p) for p in pcm) / 2 / tts.sample_rate
                print(f"  {path}  {seconds:.1f}s")

    for backend in backends.values():
        if hasattr(backend, "close"):
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
