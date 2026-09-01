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
import sys
import uuid
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
from lad_translate.session.room import TranslationRoom

log = get_logger("serve_session")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default="demo-room")
    ap.add_argument("--targets", default="fr,ar")
    ap.add_argument("--event", default="Live speaker test")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--emit-interval", type=float, default=3.0)
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument(
        "--speech-rms",
        type=float,
        default=WhisperSttAdapter.LIVE_SPEECH_RMS,
        help=(
            "refuse to transcribe a buffer quieter than this. The default suits "
            "a live microphone in a room, where background noise below it would "
            "otherwise reach the model and come back as invented words. Lower it "
            "for a quiet speaker or a recorded source, and confirm it from the "
            "'transcribing buffer' rms values logged at DEBUG"
        ),
    )
    ap.add_argument("--wait", type=float, default=900.0,
                    help="seconds to wait for a speaker before giving up")
    args = ap.parse_args()

    configure()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    db_url = os.environ["LAD_DATABASE_URL"]

    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
    row = await pool.fetchrow(
        "SELECT id::text, schema_name FROM lad_dev.tenants WHERE slug='techiemaya'"
    )
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

    async with WhisperSttAdapter(
        model_size=args.model,
        emit_interval=args.emit_interval,
        max_window_s=args.window,
        speech_rms=args.speech_rms,
    ) as stt, tts:
        session = TranslationSession(
            config=config, room=room, stt=stt, mt=mt, tts=tts, store=store, max_lag_s=3.0
        )
        outcome = await session.run()

    print(f"\n  status  {outcome.status}  chunks={outcome.chunks}")
    print(f"  backlog {outcome.backlog}")
    mt.close()
    await pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
