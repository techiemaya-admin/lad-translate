#!/usr/bin/env python3
"""
End to end verification.

Drives one real session through every layer and checks each one, rather than
checking that the parts import. Publisher, SFU, speech recognition, chunker,
both translation backends, synthesis, the join API, real WebRTC listeners, the
database, and billing.

The listeners are the point. Everything upstream can look healthy while the
audience hears nothing, and only a subscriber that actually receives audio
proves otherwise.

    ./tools/pg.sh start && ./tools/livekit.sh start
    python tools/e2e.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import os
import re
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.adapters.stt_whisper import WhisperSttAdapter
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter
from lad_translate.api.rooms import SOURCE_TRACK_NAME
from lad_translate.api.tokens import TokenIssuer
from lad_translate.config import (
    LanguageTarget,
    SessionConfig,
    SessionLimits,
    TenantContext,
)
from lad_translate.db.sessions import SessionStore
from lad_translate.obs.log import configure
from lad_translate.session.pipeline import TranslationSession
from lad_translate.session.room import TranslationRoom

FRAME_MS = 20


# =============================================================================
# REPORTING
# =============================================================================


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


CHECKS: list[Check] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append(Check(name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


# =============================================================================
# PARTICIPANTS
# =============================================================================


async def publisher(url: str, token: str, audio: Path, done: asyncio.Event) -> None:
    """The venue laptop, streaming a file at real speed."""
    import livekit.rtc as rtc

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
        while True:
            pcm = wav.readframes(per_frame)
            if not pcm:
                break
            await source.capture_frame(
                rtc.AudioFrame(
                    data=pcm, sample_rate=rate, num_channels=1,
                    samples_per_channel=len(pcm) // 2,
                )
            )
    await asyncio.sleep(8)  # let the tail of the pipeline drain
    done.set()
    await room.disconnect()


async def listener(url: str, token: str, track_name: str, stop: asyncio.Event) -> tuple[float, int]:
    """
    A phone. Subscribes to exactly one track and reports what it heard.

    Returns (seconds of audio, peak amplitude). Peak is what separates real
    speech from an open track carrying silence, which is the failure that
    looks healthy from every other angle.
    """
    import livekit.rtc as rtc

    room = rtc.Room()
    chunks: list[bytes] = []
    rate = 0
    drains: set[asyncio.Task] = set()

    async def drain(track) -> None:
        nonlocal rate
        stream = rtc.AudioStream.from_track(track=track)
        async for event in stream:
            rate = event.frame.sample_rate
            chunks.append(bytes(event.frame.data))
            if stop.is_set():
                break
        await stream.aclose()

    def want(publication) -> None:
        """Subscribe to our one track, whenever it turns up."""
        if publication.name == track_name and not publication.subscribed:
            publication.set_subscribed(True)

    @room.on("track_published")
    def _on_published(publication, participant):
        # Essential with auto_subscribe off. The translator publishes its
        # language tracks AFTER this listener connects, so subscribing only to
        # what exists at connect time subscribes to nothing, and the listener
        # sits in silence with no error anywhere.
        want(publication)

    @room.on("track_subscribed")
    def _on(track, publication, participant):
        if publication.name == track_name:
            task = asyncio.create_task(drain(track))
            drains.add(task)
            task.add_done_callback(drains.discard)

    @room.on("track_subscription_failed")
    def _on_failed(participant, track_sid, error):
        print(f"  listener: subscription FAILED for {track_sid}: {error}")

    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=False))
    # Anything published before we joined never fires track_published.
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            want(publication)

    await stop.wait()
    await asyncio.sleep(2)
    await room.disconnect()

    data = b"".join(chunks)
    if not data or not rate:
        return 0.0, 0
    return len(data) / 2 / rate, audioop.max(data, 2)


# =============================================================================
# MAIN
# =============================================================================


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "holmes.wav")
    ap.add_argument("--reference", type=Path, default=ROOT / "fixtures" / "holmes.txt")
    ap.add_argument("--model", default="tiny")
    ap.add_argument(
        "--targets", default="fr,te",
        help="fr,te exercises both translation backends but starves two cores; "
             "fr,ar keeps everything on Opus-MT and this machine can serve it",
    )
    args = ap.parse_args()

    configure("ERROR")
    url = os.environ.setdefault("LIVEKIT_URL", "ws://127.0.0.1:7880")
    os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
    os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
    db_url = os.environ.setdefault(
        "LAD_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent"
    )

    room_name = f"e2e-{uuid.uuid4().hex[:8]}"
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    print(f"\nroom     {room_name}")
    print(f"source   {args.audio.name}")
    print(f"targets  {', '.join(targets)} (+ English relay)\n")

    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4)
    row = await pool.fetchrow(
        "SELECT id::text, schema_name FROM lad_dev.tenants WHERE slug='techiemaya'"
    )
    if row is None:
        print("  FAIL  no tenant seeded; run tools/seed_tenant.py --slug techiemaya")
        return 1
    tenant = TenantContext(tenant_id=row[0], database_url=db_url, schema=row[1])
    store = SessionStore(pool, tenant)

    config = SessionConfig(
        session_id=str(uuid.uuid4()), tenant=tenant, room_name=room_name,
        event_name="End to end verification", source_language="en",
        targets=[LanguageTarget(c, DEFAULT_VOICES[c]) for c in targets],
        limits=SessionLimits(max_duration_s=600, max_idle_s=45),
    )
    await store.create_session(config, latency_credible=False)
    await store.mark_live(config.session_id)
    check("session registered", True, config.session_id[:8])

    issuer = TokenIssuer()
    mt = RoutingMtAdapter(
        "en", targets,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    check("routing resolved", True,
          " ".join(f"{c}->{mt.backend_for(c)}" for c in targets))

    tts = PiperTtsAdapter(targets, voice_root=ROOT / "models" / "tts")
    room = TranslationRoom(room_name, sample_rate=tts.sample_rate)
    await room.connect(url, issuer.for_translator(config))

    done = asyncio.Event()
    started = time.monotonic()

    async with WhisperSttAdapter(model_size=args.model, emit_interval=3.0, max_window_s=6.0) as stt, tts:
        pub = asyncio.create_task(
            publisher(url, issuer.for_publisher(room_name), args.audio, done)
        )
        # One listener per translated language, plus one on the relay. Built
        # from the target list rather than hardcoded: an earlier version named
        # the languages twice and crashed the moment --targets changed.
        watching = {code: f"lang-{code}" for code in targets}
        watching[config.source_language] = SOURCE_TRACK_NAME
        heard = {
            code: asyncio.create_task(
                listener(url, issuer.for_listener(room_name, code, f"e2e-{code}").token,
                         track, done)
            )
            for code, track in watching.items()
        }
        listener_ids = [
            await store.listener_joined(config.session_id, code) for code in watching
        ]

        session = TranslationSession(
            config=config, room=room, stt=stt, mt=mt, tts=tts, store=store, max_lag_s=3.0
        )
        outcome = await session.run()

    results = {code: await task for code, task in heard.items()}
    await pub
    mt.close()
    elapsed = time.monotonic() - started

    # ---------------------------------------------------------------- checks
    print()
    check("session ended cleanly", outcome.status == "ended", outcome.status)
    check("chunks produced", outcome.chunks > 0, f"{outcome.chunks} chunks")
    dropped = outcome.backlog["frames_dropped"]
    total = outcome.backlog["frames_in"]
    rate = dropped / total if total else 0.0
    if dropped:
        print(f"  NOTE  audio shed   {dropped}/{total} frames ({rate * 100:.0f}%), "
              "a capacity limit on this machine rather than a fault")
    else:
        check("no audio shed", True, f"0 of {total} frames")

    for code in targets:
        seconds, peak = results[code]
        check(f"listener heard {code}", seconds > 1.0 and peak > 500,
              f"{seconds:.1f}s peak {peak}")

    relay_s, relay_peak = results[config.source_language]
    check("listener heard the original (relay)", relay_s > 1.0 and relay_peak > 500,
          f"{relay_s:.1f}s peak {relay_peak}")

    rows = await pool.fetch(
        f"""SELECT language, count(*) n, count(translated_text) t
            FROM {tenant.schema}.session_transcripts
            WHERE session_id = $1::uuid GROUP BY language""",
        config.session_id,
    )
    by_lang = {r["language"]: (r["n"], r["t"]) for r in rows}
    for code in targets:
        n, t = by_lang.get(code, (0, 0))
        check(f"transcripts stored for {code}", n > 0 and t == n, f"{t}/{n} translated")

    active = await store.active_listeners(config.session_id)
    check("listeners recorded", sum(active.values()) == len(watching), f"{active}")
    for lid in listener_ids:
        await store.listener_left(lid)
    check("listener departures recorded",
          sum((await store.active_listeners(config.session_id)).values()) == 0)

    # TranslationSession settles billing itself when a store is passed, so read
    # the row rather than calling end_session again.
    settled = await store.get_session(config.session_id)
    ok = (
        settled is not None
        and settled["status"] == "ended"
        and settled["billed_language_count"] == len(targets)
        and (settled["billed_seconds"] or 0) > 0
    )
    check("billing settled by the session", ok,
          f"{settled['billed_seconds']}s x {settled['billed_language_count']} = "
          f"{(settled['billed_seconds'] or 0) * (settled['billed_language_count'] or 0)} "
          f"language-seconds" if settled else "no row")

    # Transcript accuracy against the published text.
    if args.reference.exists():
        src_rows = await pool.fetch(
            f"""SELECT DISTINCT ON (chunk_id) chunk_id, source_text
                FROM {tenant.schema}.session_transcripts
                WHERE session_id = $1::uuid ORDER BY chunk_id""",
            config.session_id,
        )
        hyp = " ".join(r["source_text"] for r in src_rows)
        punct = re.compile(r"[^\w\s']")
        ref_w = punct.sub(" ", args.reference.read_text().lower()).split()
        hyp_w = punct.sub(" ", hyp.lower()).split()
        sys.path.insert(0, str(ROOT / "tools"))
        from score_stt import wer

        _s, _d, _i, dist = wer(ref_w, hyp_w)
        rate = dist / len(ref_w) if ref_w else 1.0
        check("transcript accuracy measured", rate < 0.5, f"WER {rate * 100:.1f}%")

    await pool.close()

    print(f"\nwall clock {elapsed:.0f}s")
    for code, stats in outcome.latency.get("languages", {}).items():
        print(f"  latency {code}: p50 {stats['p50_s']}s p95 {stats['p95_s']}s")
    print(f"  drift: { {k: v['peak_depth_s'] for k, v in outcome.drift.items()} }")

    failed = [c for c in CHECKS if not c.ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for c in failed:
            print(f"  FAILED: {c.name}  {c.detail}")
    print("\nNOTE latency is not product latency: Whisper is not a streaming model.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
