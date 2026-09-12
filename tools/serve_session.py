#!/usr/bin/env python3
"""
Run a translation session and wait for a speaker.

This is the realistic shape: the service runs, and the venue publishes when it
is ready. tools/session_live.py plays a file and is for measurement; this one
sits waiting for whatever publishes source-audio, whether that is a desk feed
or someone's phone on /speak.

    ./tools/demo.sh down            # stop any file-driven session
    python tools/serve_session.py --room demo-room --targets fr,ar

Runs until the source track ends or the idle cap is reached.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter
from lad_translate.api.tokens import TokenIssuer
from lad_translate.config import (
    LanguageTarget,
    SessionConfig,
    SessionLimits,
    TenantContext,
)
from lad_translate.db.pool import control_schema
from lad_translate.db.sessions import SessionStore
from lad_translate.obs.log import configure, get_logger
from lad_translate.session.pipeline import TranslationSession
from lad_translate.session.recording import RecordingSink
from lad_translate.session.room import TranslationRoom
from lad_translate.session.sinks import FanOutSink

log = get_logger("serve_session")


def build_stt_backend(args):
    """
    Construct the STT backend named by --stt.

    The two backends take different options because they are different shapes:
    Whisper needs a window and an emit interval because it re-transcribes a
    sliding buffer, and a streaming transducer has neither - it encodes each
    step once and carries its context in a cache tensor. Passing Whisper's
    knobs to it would be meaningless rather than merely unused.
    """
    from lad_translate.adapters.registry import build_stt

    if args.stt == "fastconformer":
        return build_stt(
            "fastconformer",
            lookahead=args.lookahead,
            device=args.device if args.device != "cpu" else None,
            vad=getattr(args, "vad", True),
        )
    return build_stt(
        "faster-whisper",
        model_size=args.model,
        device=args.device,
        cpu_threads=getattr(args, "cpu_threads", 0),
        emit_interval=args.emit_interval,
        max_window_s=args.window,
    )



async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default="demo-room")
    ap.add_argument("--tenant", default="techiemaya", help="tenant slug in the control schema")
    ap.add_argument("--stt", default=os.getenv("STT_BACKEND", "faster-whisper"),
                    choices=["faster-whisper", "fastconformer"])
    ap.add_argument("--lookahead", default="480ms",
                    help="fastconformer only: 0ms 80ms 480ms 1040ms")
    # Default on. Off is what this adapter shipped as, and it made words out of
    # room tone; --no-vad exists to measure against a clean file, not for a room.
    ap.add_argument("--vad", dest="vad", action="store_true", default=True)
    ap.add_argument("--no-vad", dest="vad", action="store_false")
    ap.add_argument("--targets", default="fr,ar")
    ap.add_argument("--event", default="Live speaker test")
    ap.add_argument("--model", default="tiny")
    # Defaults to cpu so nothing about running this on the dev Mac changes.
    # The GPU box passes --device cuda, which is the whole reason it exists:
    # without it faster-whisper loads int8 on CPU and the L4 sits idle.
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # 0 lets ctranslate2 decide, which makes latency a property of the machine
    # rather than of the config. On a 32 vCPU box its default runs 4x slower
    # than 4 threads does - see adapters/stt_whisper.py:cpu_threads.
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--emit-interval", type=float, default=3.0)
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--record-dir", type=Path, default=None,
                    help="where recordings go; without it the session cannot record at all")
    ap.add_argument("--record", action="store_true",
                    help="start recording as soon as the session starts (needs --record-dir)")
    ap.add_argument("--wait", type=float, default=900.0,
                    help="seconds to wait for a speaker before giving up")
    args = ap.parse_args()

    configure()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    db_url = os.environ["LAD_DATABASE_URL"]

    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
    # Was: a literal lad_dev.tenants and a literal slug. Both were wrong the
    # moment this ran anywhere real - lad_dev is Mr LAD's control schema and
    # its tenants table has neither schema_name nor is_active, so the query
    # died with UndefinedColumnError on the first session. lad-translate owns
    # its own control schema; see deploy/README.md.
    control = control_schema()
    row = await pool.fetchrow(
        f"SELECT id::text, schema_name FROM {control}.tenants WHERE slug = $1 AND is_active",
        args.tenant,
    )
    if row is None:
        log.error(
            "tenant not found; seed it with tools/seed_tenant.py",
            extra={"tenant": args.tenant, "control_schema": control},
        )
        return 1
    tenant = TenantContext(tenant_id=row[0], database_url=db_url, schema=row[1])
    store = SessionStore(pool, tenant)

    # End anything already live in this room, so the join page offers exactly
    # one session and the audience cannot land on a dead one.
    for existing in await store.live_sessions():
        full = await store.get_session(existing["session_id"])
        if full and full["room_name"] == args.room:
            await store.end_session(existing["session_id"], "superseded")

    config = SessionConfig(
        session_id=str(uuid.uuid4()), tenant=tenant, room_name=args.room,
        event_name=args.event, source_language="en",
        targets=[LanguageTarget(c, DEFAULT_VOICES[c]) for c in targets],
        # Generous idle cap: a person needs time to scan a code and start
        # talking, and the session must not expire while they do.
        limits=SessionLimits(max_duration_s=3 * 3600, max_idle_s=args.wait),
    )
    await store.create_session(config, latency_credible=False)
    await store.mark_live(config.session_id)

    issuer = TokenIssuer()
    mt = RoutingMtAdapter(
        "en", targets,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    tts = PiperTtsAdapter(targets, voice_root=ROOT / "models" / "tts")
    # --wait governs BOTH the wait for a speaker and the idle cap. Setting
    # only the idle cap left the room giving up after its 60s default, which
    # is what killed the first live attempt.
    room = TranslationRoom(
        args.room, sample_rate=tts.sample_rate, source_timeout_s=args.wait
    )
    await room.connect(issuer.internal_url, issuer.for_translator(config))

    print(f"\n  session   {config.session_id}")
    print(f"  room      {args.room}")
    print(f"  languages {', '.join(targets)}")
    print(f"\n  speak     /speak/{config.session_id}")
    print(f"  listen    /s/{config.session_id}")
    print("\n  waiting for a speaker...\n", flush=True)

    # A recorder is installed whenever there is somewhere to put files, armed
    # or not, so the console can start a recording mid-talk with a signal
    # rather than a restart. SIGUSR1 starts, SIGUSR2 stops; both idempotent,
    # so the console can send "be recording" without knowing the state.
    recorder = None
    sink = None
    if args.record_dir is not None:
        recorder = RecordingSink(args.record_dir, config.session_id, args.room, args.event)
        sink = FanOutSink(room, recorder)

    async with build_stt_backend(args) as stt, tts:
        session = TranslationSession(
            config=config, room=room, stt=stt, mt=mt, tts=tts, store=store, max_lag_s=3.0,
            sink=sink, recorder=recorder,
        )
        if recorder is not None:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGUSR1, session.start_recording)
            loop.add_signal_handler(signal.SIGUSR2, session.stop_recording)
            if args.record:
                session.start_recording()
        outcome = await session.run()

    print(f"\n  status  {outcome.status}  chunks={outcome.chunks}")
    print(f"  backlog {outcome.backlog}")
    mt.close()
    await pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
