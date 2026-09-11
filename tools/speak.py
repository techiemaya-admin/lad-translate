#!/usr/bin/env python3
"""
Publish a WAV into a room as the speaker, the way the speaker page does.

The counterpart of tools/listen.py. Joins through the real join API and
publishes the file at real speed on the source track, so everything
downstream - the session, the listeners, an output agent - sees exactly what
a phone at the lectern would produce. Written for checking a venue rig
without asking anyone to talk into it for ninety seconds.

    python tools/speak.py --room dubai-demo --base https://lad-translate-dev-...run.app
    python tools/speak.py --room dubai-demo --audio fixtures/keynote.wav --loop
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import ssl
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.obs.log import configure

FRAME_MS = 20


def speaker_grant(base: str, room: str, insecure: bool) -> dict:
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(f"{base}/api/rooms/{room}/speak", data=b"", method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
        return json.load(response)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True)
    ap.add_argument("--base", required=True, help="join service base URL")
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "holmes.wav")
    ap.add_argument("--loop", action="store_true", help="play the file again when it ends")
    ap.add_argument("--insecure", action="store_true", help="accept a dev TLS certificate")
    args = ap.parse_args()

    configure("ERROR")
    import livekit.rtc as rtc

    grant = speaker_grant(args.base, args.room, args.insecure)
    print(f"  speaking into {args.room}  ({grant['event_name']}, {grant['source_language']})")

    room = rtc.Room()
    await room.connect(grant["url"], grant["token"])
    with wave.open(str(args.audio), "rb") as wav:
        rate = wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            print("  the file must be 16-bit mono", file=sys.stderr)
            return 2
        per_frame = int(rate * FRAME_MS / 1000)
        source = rtc.AudioSource(rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(grant["track_name"], source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"  publishing {args.audio.name} @ {rate} Hz" + (" on a loop" if args.loop else ""))
        try:
            while True:
                pcm = wav.readframes(per_frame)
                if not pcm:
                    if not args.loop:
                        break
                    wav.rewind()
                    continue
                await source.capture_frame(
                    rtc.AudioFrame(
                        data=pcm, sample_rate=rate, num_channels=1,
                        samples_per_channel=len(pcm) // 2,
                    )
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    # Let the tail of the pipeline drain before the source track disappears.
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(8)
    await room.disconnect()
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
