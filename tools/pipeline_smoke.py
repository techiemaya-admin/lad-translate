#!/usr/bin/env python3
"""
End to end smoke test for the local stack.

Audio file in, translated speech out, one WAV per target language, with the
latency breakdown printed. No LiveKit and no database: this exercises the four
stages that do the work.

    audio -> STT -> phrase chunker -> translation fan-out -> TTS

USE THIS TO READ TRANSLATIONS, NOT TO MEASURE LATENCY.

This tool synthesises every language inline, in the same loop that consumes STT
hypotheses. While TTS runs the frame consumption stalls, so audio backs up
behind the guard and gets shed: two languages at roughly half a second each,
across a dozen chunks, stalled about 12 seconds of a 45 second clip and shed
23% of it. That is this tool's structure, not the pipeline's.

session/pipeline.py does not work this way. It runs a worker task per language,
so synthesis never blocks the source. For latency or drop-rate figures use
tools/session_live.py, which drives the real TranslationSession.

The STT is also faster-whisper here, which is not a streaming model, so even
session_live's latency is not product latency. See adapters/stt_whisper.py.

Usage:
    python tools/pipeline_smoke.py --audio fixtures/keynote.wav --targets fr,de
    python tools/pipeline_smoke.py --audio fixtures/keynote.wav --targets fr --realtime
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.base import AudioFrame, VoiceSpec
from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.adapters.stt_whisper import WhisperSttAdapter
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter
from lad_translate.chunker import ChunkerConfig, PhraseChunker
from lad_translate.obs.latency import LatencyRecorder, Stage
from lad_translate.obs.log import configure, get_logger
from lad_translate.session.backpressure import BacklogGuard, guarded

log = get_logger("smoke")

FRAME_MS = 100


async def read_frames(path: Path, realtime: bool) -> AsyncIterator[AudioFrame]:
    """Stream a WAV as AudioFrames, optionally paced at real speed."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        per_frame = int(rate * FRAME_MS / 1000)
        t_audio = 0.0
        start = time.monotonic()
        while True:
            pcm = wav.readframes(per_frame)
            if not pcm:
                break
            if realtime:
                # Hold the frame back until its moment, so t_wall reflects a
                # live stream rather than how fast the disk reads.
                due = start + t_audio
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield AudioFrame(pcm=pcm, sample_rate=rate, t_audio=t_audio, t_wall=time.monotonic())
            t_audio += len(pcm) / 2 / rate


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "keynote.wav")
    ap.add_argument("--targets", default="fr", help="comma separated language codes")
    ap.add_argument("--model", default="tiny", help="whisper model size")
    ap.add_argument("--realtime", action="store_true", help="pace audio at real speed")
    ap.add_argument("--emit-interval", type=float, default=1.0,
                    help="must exceed window * RTF or the backlog grows without bound")
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=ROOT / "fixtures" / "out")
    ap.add_argument("--max-lag", type=float, default=3.0,
                    help="drop older audio once the backlog passes this (0 disables)")
    args = ap.parse_args()

    configure("WARNING")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\naudio    {args.audio.name}")
    print(f"targets  {', '.join(targets)}")
    print(f"pacing   {'real time' if args.realtime else 'as fast as possible'}\n")

    # Per language: Indic to NLLB, the rest to Opus-MT. See mt_routing.py.
    mt = RoutingMtAdapter(
        "en", targets,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    tts = PiperTtsAdapter(targets, voice_root=ROOT / "models" / "tts")
    chunker = PhraseChunker(ChunkerConfig(min_words=4, max_words=18, max_wait_s=1.0))
    recorder = LatencyRecorder(slo_seconds=2.0)

    audio_out: dict[str, list[bytes]] = {t: [] for t in targets}
    all_final = True
    transcript: list[str] = []
    started = time.monotonic()

    guard = BacklogGuard(max_lag_s=args.max_lag) if args.max_lag > 0 else None

    async with WhisperSttAdapter(model_size=args.model, emit_interval=args.emit_interval,
                          max_window_s=args.window) as stt, tts:
        frames = read_frames(args.audio, args.realtime)
        if guard is not None:
            frames = guarded(frames, guard)
        async for hyp in stt.transcribe(frames):
            if not recorder.clock.anchored:
                recorder.clock.anchor(t_audio=0.0, t_wall=started)

            for chunk in chunker.feed(hyp):
                if chunk.reason.value != "final":
                    all_final = False
                transcript.append(chunk.text)
                print(f"[{chunk.reason.value:<18} lag {chunk.stability_lag:4.2f}s] {chunk.text}")

                for lang in targets:
                    recorder.open_chunk(chunk.chunk_id, lang, chunk.t_audio_end)
                    recorder.mark(chunk.chunk_id, lang, Stage.COMMITTED, chunk.t_wall_committed)

                translations = await mt.translate_many(chunk.text, "en", targets)
                now = time.monotonic()
                for lang, text in translations.items():
                    recorder.mark(chunk.chunk_id, lang, Stage.TRANSLATED, now)
                    print(f"    {lang}: {text}")

                for lang, text in translations.items():
                    if not text.strip():
                        continue
                    voice = VoiceSpec(language=lang, voice_id=DEFAULT_VOICES[lang])
                    first = True
                    async for speech in tts.synthesise(text, voice, chunk.chunk_id):
                        if first:
                            recorder.mark(chunk.chunk_id, lang, Stage.TTS_FIRST_AUDIO, speech.t_wall)
                            recorder.mark(chunk.chunk_id, lang, Stage.PUBLISHED, time.monotonic())
                            first = False
                        audio_out[lang].append(speech.pcm)
                print()

        for chunk in chunker.flush():
            transcript.append(chunk.text)
            print(f"[flush] {chunk.text}")

    for lang, pieces in audio_out.items():
        if not pieces:
            continue
        path = args.out / f"keynote-{lang}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(tts.sample_rate)
            wav.writeframes(b"".join(pieces))
        seconds = sum(len(p) for p in pieces) / 2 / tts.sample_rate
        print(f"wrote {path}  {seconds:.1f}s")

    print(f"\nwall clock: {time.monotonic() - started:.1f}s")
    if guard is not None:
        print(f"backlog: {guard.summary()}")
    print("\ntranscript:")
    print(" ", " ".join(transcript))

    # Latency only means anything when the audio was fed at real speed. Without
    # --realtime the file is read as fast as the disk allows, so the audio clock
    # and the wall clock diverge and every figure is fiction. Refuse to print
    # rather than print something misleading.
    if args.realtime:
        summary = recorder.summary()
        print("\nlatency (NOT credible as product latency: Whisper is not streaming)")
        for lang, stats in summary["languages"].items():
            print(f"  {lang}: p50 {stats['p50_s']}s  p95 {stats['p95_s']}s  stages {stats['stages']}")
    else:
        print("\nlatency not reported: run with --realtime for meaningful timings")

    if all_final:
        print(
            "\nNOTE  Every chunk was committed on a FINAL hypothesis, so the\n"
            "      chunker degenerated to a pass-through of the STT's own\n"
            "      utterance segmentation. Its stability and clause logic only\n"
            "      earns anything against a backend that emits genuine interims\n"
            "      mid-utterance. Another reason Whisper is the wrong backend."
        )
    mt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
