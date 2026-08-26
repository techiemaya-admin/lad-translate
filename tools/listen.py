#!/usr/bin/env python3
"""
Listen to a language from the command line, the way a phone would.

Joins through the real join API rather than minting a token directly, so this
exercises the same path an audience member takes: resolve the room, get a
token, subscribe to exactly one track. If this hears audio, a phone will too.

Written for the venue check: confirming a language is actually on air without
borrowing someone's handset, and without the certificate dance that a browser
requires.

    python tools/listen.py --room demo-room --language te --seconds 30
    python tools/listen.py --room demo-room --language en --out original.wav
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import contextlib
import json
import ssl
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.api.tokens import TokenIssuer
from lad_translate.obs.log import configure


def _permissive_ssl() -> ssl.SSLContext:
    """
    Accept the dev CA, for the case where --base points at the proxy.

    The default does not: local tooling talks to the join service directly on
    http, because the TLS proxy exists for browsers that will not touch a
    microphone over plain http. Routing a local process through it buys
    nothing, and Caddy only holds a certificate for the LAN address anyway, so
    https://127.0.0.1:8443 fails the handshake outright.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def join(base: str, room: str, language: str) -> dict:
    """Ask the join API for a token, exactly as the browser page does."""
    ctx = _permissive_ssl()

    req = urllib.request.Request(
        f"{base}/api/rooms/{room}/join",
        data=json.dumps({"language": language}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
        return json.load(response)


def leave(base: str, listener_id: str, session_id: str) -> None:
    ctx = _permissive_ssl()
    req = urllib.request.Request(
        f"{base}/api/listeners/{listener_id}/leave?session_id={session_id}",
        data=b"", method="POST",
    )
    with contextlib.suppress(Exception):
        urllib.request.urlopen(req, context=ctx, timeout=10).close()


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default="demo-room")
    ap.add_argument("--language", default="te")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument(
        "--base", default="http://127.0.0.1:8080",
        help="join service base URL; the local service directly, not via the "
             "TLS proxy, which only serves the LAN address",
    )
    ap.add_argument("--out", type=Path, help="write what was heard to a WAV")
    args = ap.parse_args()

    configure("ERROR")
    import livekit.rtc as rtc

    grant = join(args.base, args.room, args.language)
    print(f"  joined {args.room} as {args.language}")
    print(f"  track  {grant['track_name']}")

    # The token advertises the public wss URL for browsers. This process runs
    # on the host and dials LiveKit directly: the SDK does not trust the dev CA.
    issuer = TokenIssuer()
    url = issuer.internal_url

    room = rtc.Room()
    chunks: list[bytes] = []
    rate = 0
    drains: set[asyncio.Task] = set()
    attached = asyncio.Event()
    want_name = grant["track_name"]

    def want(publication) -> None:
        if publication.name == want_name and not publication.subscribed:
            publication.set_subscribed(True)

    @room.on("track_published")
    def _on_published(publication, participant):
        want(publication)

    @room.on("track_subscribed")
    def _on(track, publication, participant):
        if publication.name != want_name:
            return
        attached.set()

        async def drain() -> None:
            nonlocal rate
            stream = rtc.AudioStream.from_track(track=track)
            async for event in stream:
                rate = event.frame.sample_rate
                chunks.append(bytes(event.frame.data))
            await stream.aclose()

        task = asyncio.create_task(drain())
        drains.add(task)
        task.add_done_callback(drains.discard)

    await room.connect(url, grant["token"], rtc.RoomOptions(auto_subscribe=False))
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            want(publication)

    try:
        await asyncio.wait_for(attached.wait(), timeout=min(30.0, args.seconds))
        print(f"  attached, listening for {args.seconds:.0f}s...")
    except TimeoutError:
        print(f"  never received {want_name}: nothing is publishing it")

    await asyncio.sleep(args.seconds)
    await room.disconnect()
    leave(args.base, grant["listener_id"], grant["session_id"])

    data = b"".join(chunks)
    if not data or not rate:
        print("  heard nothing")
        return 1

    seconds = len(data) / 2 / rate
    peak = audioop.max(data, 2)
    level = audioop.rms(data, 2)
    # Peak, not duration. An open track carrying silence has plenty of
    # duration, and that is the failure that looks healthy from every angle.
    print(f"  heard {seconds:.1f}s @ {rate}Hz  rms {level}  peak {peak}  "
          f"{'SPEECH' if peak > 500 else 'SILENT'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(args.out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(data)
        print(f"  wrote {args.out}")
    return 0 if peak > 500 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
