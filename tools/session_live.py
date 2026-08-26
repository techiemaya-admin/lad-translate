#!/usr/bin/env python3
"""
Full session through a real self-hosted LiveKit room.

Runs three participants in one process:

    publisher   stands in for the venue laptop, streams a WAV as source-audio
    translator  the TranslationSession under test
    listener    subscribes to one language track and records what it hears

The listener is the part that matters. Everything upstream can look healthy
while the audience hears nothing, and only a real subscriber proves otherwise.

Start the server first:

    ./tools/livekit.sh start
    export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret

Usage:
    python tools/session_live.py --targets fr
    python tools/session_live.py --targets fr,de --audio fixtures/keynote.wav
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
import wave
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.adapters.stt_whisper import WhisperSttAdapter
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter
from lad_translate.api.tokens import TokenIssuer
from lad_translate.config import (
    LanguageTarget,
    SessionConfig,
    SessionLimits,
    TenantContext,
)
from lad_translate.db.sessions import SessionStore
from lad_translate.obs.log import configure, get_logger
from lad_translate.session.pipeline import TranslationSession
from lad_translate.session.room import SOURCE_TRACK_NAME, TranslationRoom

log = get_logger("session_live")
FRAME_MS = 20


async def venue_publisher(
    url: str, token: str, audio: Path, done: asyncio.Event, loop: bool = False
) -> None:
    """Stream a WAV into the room at real speed, as the venue desk would."""
    from livekit import rtc

    room = rtc.Room()
    await room.connect(url, token)
    with wave.open(str(audio), "rb") as wav:
        rate = wav.getframerate()
        per_frame = int(rate * FRAME_MS / 1000)
        source = rtc.AudioSource(rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(SOURCE_TRACK_NAME, source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"publisher  streaming {audio.name} at {rate}Hz"
              + (" (looping)" if loop else ""))
        passes = 0
        while True:
            pcm = wav.readframes(per_frame)
            if not pcm:
                if not loop:
                    break
                # Rewind for a demo that runs long enough to join and listen
                # to. A speaker does not restart, so this is for demos only,
                # never for measuring anything.
                passes += 1
                print(f"publisher  restarting clip (pass {passes + 1})")
                wav.rewind()
                continue
            await source.capture_frame(
                rtc.AudioFrame(
                    data=pcm, sample_rate=rate, num_channels=1,
                    samples_per_channel=len(pcm) // 2,
                )
            )
    print("publisher  source audio finished")
    # Hold the room open briefly so the tail of the pipeline can drain.
    await asyncio.sleep(5)
    done.set()
    await room.disconnect()


async def listener(url: str, token: str, language: str, out: Path, stop: asyncio.Event) -> float:
    """Subscribe to one language track and write what actually arrives."""
    from livekit import rtc

    room = rtc.Room()
    received: list[bytes] = []
    sample_rate = 0
    attached = asyncio.Event()
    # Hold a reference. A task with no strong reference can be garbage
    # collected while still running, which here would stop the listener
    # receiving audio with nothing logged to say why.
    drains: set[asyncio.Task] = set()

    @room.on("track_subscribed")
    def _on(track, publication, participant):
        if publication.name == f"lang-{language}":
            attached.set()
            task = asyncio.create_task(_drain(track))
            drains.add(task)
            task.add_done_callback(drains.discard)

    async def _drain(track) -> None:
        nonlocal sample_rate
        stream = rtc.AudioStream.from_track(track=track)
        async for event in stream:
            frame = event.frame
            sample_rate = frame.sample_rate
            received.append(bytes(frame.data))
            if stop.is_set():
                break
        await stream.aclose()

    await room.connect(url, token)
    print(f"listener   waiting for lang-{language}")
    try:
        await asyncio.wait_for(attached.wait(), timeout=90)
        print(f"listener   attached to lang-{language}")
    except TimeoutError:
        print(f"listener   NEVER received lang-{language}")
    await stop.wait()
    await asyncio.sleep(2)
    await room.disconnect()

    if not received or not sample_rate:
        return 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(received))
    seconds = sum(len(c) for c in received) / 2 / sample_rate
    print(f"listener   wrote {out} ({seconds:.1f}s @ {sample_rate}Hz)")
    return seconds


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "keynote.wav")
    ap.add_argument("--targets", default="fr")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--emit-interval", type=float, default=3.0,
                    help="must exceed window_s * RTF or the backlog grows without bound")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--room", default=f"lad-{uuid.uuid4().hex[:8]}",
                    help="use an existing session's room to listen from a browser")
    ap.add_argument("--no-store", action="store_true",
                    help="skip persisting transcripts (default is to store them)")
    ap.add_argument("--loop", action="store_true",
                    help="repeat the clip so there is time to join and listen")
    ap.add_argument("--out", type=Path, default=ROOT / "fixtures" / "heard")
    args = ap.parse_args()

    configure("WARNING")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    issuer = TokenIssuer()
    url = issuer.livekit_url

    config = SessionConfig(
        session_id=str(uuid.uuid4()),
        tenant=TenantContext(
            tenant_id=str(uuid.uuid4()),
            database_url="postgresql://unused",
            schema="tenant_live",
        ),
        room_name=args.room,
        event_name="Live smoke",
        source_language="en",
        targets=[LanguageTarget(c, DEFAULT_VOICES[c]) for c in targets],
        limits=SessionLimits(max_duration_s=600, max_idle_s=30),
    )

    print(f"\nroom       {args.room}")
    print(f"targets    {', '.join(targets)}\n")

    # Persist transcripts. The session data model exists and is tested, and
    # without this the only record of what the audience heard is the audio
    # itself, which cannot be read or diffed.
    pool = store = None
    if not args.no_store:
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                os.environ["LAD_DATABASE_URL"], min_size=1, max_size=3
            )
            row = await pool.fetchrow(
                "SELECT id::text, schema_name FROM lad_dev.tenants WHERE slug='techiemaya'"
            )
            tenant = TenantContext(
                tenant_id=row[0], database_url=os.environ["LAD_DATABASE_URL"], schema=row[1]
            )
            config = replace(config, tenant=tenant)
            store = SessionStore(pool, tenant)

            # Reuse a live session already registered for this room rather
            # than creating a second one. tools/demo.sh seeds the session so it
            # can print the join URL; creating another here would leave the
            # audience on a session row that never receives a transcript.
            reused = None
            for row in await store.live_sessions():
                full = await store.get_session(row["session_id"])
                if full and full["room_name"] == config.room_name:
                    reused = full
                    break

            if reused is not None:
                config = replace(config, session_id=reused["session_id"])
                print(f"joined existing session {config.session_id} in {tenant.schema}")
            else:
                await store.create_session(config, latency_credible=False)
                print(f"created session {config.session_id} in {tenant.schema}")
        except Exception as exc:
            print(f"not storing transcripts: {exc}")
            store = None

    published_done = asyncio.Event()
    tts = PiperTtsAdapter(targets, voice_root=ROOT / "models" / "tts")
    # Per language: Indic to NLLB, the rest to Opus-MT. See mt_routing.py.
    mt = RoutingMtAdapter(
        "en", targets,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    room = TranslationRoom(args.room, sample_rate=tts.sample_rate)
    await room.connect(url, issuer.for_translator(config))


    started = time.monotonic()
    # Load the STT model BEFORE the publisher starts.
    #
    # Whisper small takes about 32 seconds to load on this machine. Starting
    # the publisher first means that much speech streams into the room with
    # nothing subscribed to catch it: the SFU does not hold it for a
    # subscriber that has not arrived, so it is gone. On a 45 second clip
    # that silently discards most of the test.
    async with WhisperSttAdapter(model_size=args.model, emit_interval=args.emit_interval, max_window_s=args.window) as stt, tts:
        pub_task = asyncio.create_task(
            venue_publisher(
                url, issuer.for_publisher(args.room), args.audio,
                published_done, loop=args.loop,
            )
        )
        listen_task = asyncio.create_task(
            listener(
                url,
                issuer.for_listener(args.room, targets[0], "smoke").token,
                targets[0],
                args.out / f"heard-{targets[0]}.wav",
                published_done,
            )
        )
        session = TranslationSession(
            config=config, room=room, stt=stt, mt=mt, tts=tts,
            store=store, max_lag_s=3.0
        )
        outcome = await session.run()

    heard = await listen_task
    await pub_task
    mt.close()
    if pool is not None:
        await pool.close()

    print(f"\nwall clock {time.monotonic() - started:.1f}s")
    print(f"status     {outcome.status}  chunks={outcome.chunks}")
    print(f"backlog    {outcome.backlog}")
    print(f"drift      {outcome.drift}")
    for lang, stats in outcome.latency.get("languages", {}).items():
        print(f"latency    {lang}: p50 {stats['p50_s']}s p95 {stats['p95_s']}s {stats['stages']}")
    print(f"\nlistener heard {heard:.1f}s of audio")
    print("NOTE latency is not credible: Whisper is not a streaming model.")
    return 0 if heard > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
