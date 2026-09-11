#!/usr/bin/env python3
"""
The venue's output agent: translated audio from the room, out as AES67.

Runs on a box on the venue's audio network - a Linux NUC or a Raspberry Pi in
the rack is enough - and does what a phone does, N times over: joins the room
as a listener for each language the channel map names, and instead of playing
what it hears, writes it into the AES67 flows the profile describes. The IR
transmitter's Dante inputs subscribe to those flows in Dante Controller.

It sits at the venue because AES67 is multicast on the local network and the
pipeline is in a cloud region. It reuses the listener path because the
listener path is the one that is proven: if a phone can hear a language, so
can this.

    python tools/output_agent.py --base https://lad-translate-dev-...run.app \\
        --room dubai-demo --profile main-hall.json \\
        --interface 192.168.10.5 --ptp-grandmaster 00-1d-c1-ff-fe-12-34-56

The profile is the device as the console saved it: open the device's page in
the console and use "Profile JSON", or GET /console/api/outputs/devices/<id>.

HOST CHECKLIST, in the order things go wrong:

  1. The box is on the Dante VLAN, on a cabled port. --interface names that
     port's address, or multicast leaves on the wrong NIC and nobody sees it.
  2. linuxptp is running and locked to the venue's PTP grandmaster:
         ptp4l -i eth0 -s --domainNumber 0     (slave to the Dante GM)
         phc2sys -s eth0 -c CLOCK_REALTIME -w   (system clock follows it)
     --ptp-grandmaster is that GM's identity (pmc -u -b 0 'GET GRANDMASTER_SETTINGS_NP'
     or Dante Controller > Clock Status). Without this the flow is valid RTP
     that a software receiver plays and a Dante device shows with a clock
     warning. The agent does not do PTP and cannot fix this for you.
  3. AES67 mode is enabled on the receiving Dante devices (Dante Controller >
     Device Config). Off by default.
  4. The multicast range is one Dante listens to: 239.69.0.0/16 by default.

The status line every ten seconds is the whole health picture: packets sent,
seconds buffered per language, and phrases dropped. Zero packets means the
socket is dead; a growing buffer means the room is ahead of the wire; drops
mean the drift controller upstream will be skipping too.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.config import OutputChannel, OutputDevice
from lad_translate.obs.log import configure, get_logger
from lad_translate.session.aes67 import Aes67Config, Aes67Sink

log = get_logger("output_agent")


def load_profile(path: Path) -> OutputDevice:
    """The console's device JSON, validated by the same rules it was saved under."""
    raw = json.loads(path.read_text())
    return OutputDevice(
        device_id=raw.get("device_id", ""),
        name=raw["name"],
        kind=raw.get("kind", "aes67"),
        device_name=raw.get("device_name", "AES67"),
        channel_count=int(raw["channel_count"]),
        sample_rate=int(raw.get("sample_rate", 48000)),
        enabled=bool(raw.get("enabled", True)),
        channels=tuple(
            OutputChannel(
                language=c["language"],
                channel=int(c["channel"]),
                ir_channel=c.get("ir_channel"),
                label=c.get("label", ""),
                gain_db=float(c.get("gain_db", 0.0)),
                enabled=bool(c.get("enabled", True)),
            )
            for c in raw.get("channels", [])
        ),
    )


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def join(base: str, room: str, language: str, ctx: ssl.SSLContext) -> dict:
    """Ask the join API for a listener token, exactly as the browser page does."""
    req = urllib.request.Request(
        f"{base}/api/rooms/{room}/join",
        data=json.dumps({"language": language}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
        return json.load(response)


def leave(base: str, listener_id: str, session_id: str, ctx: ssl.SSLContext) -> None:
    req = urllib.request.Request(
        f"{base}/api/listeners/{listener_id}/leave?session_id={session_id}",
        data=b"",
        method="POST",
    )
    with contextlib.suppress(Exception):
        urllib.request.urlopen(req, context=ctx, timeout=10).close()


async def listen(language: str, grant: dict, sink: Aes67Sink, stop: asyncio.Event) -> None:
    """One room connection, one language track, straight into the sink."""
    import livekit.rtc as rtc

    room = rtc.Room()
    want_name = grant["track_name"]
    drains: set[asyncio.Task] = set()

    def want(publication) -> None:
        if publication.name == want_name and not publication.subscribed:
            publication.set_subscribed(True)

    @room.on("track_published")
    def _on_published(publication, participant):
        want(publication)

    @room.on("track_subscribed")
    def _on_subscribed(track, publication, participant):
        if publication.name != want_name:
            return
        log.info("language track attached", extra={"language": language, "track": want_name})

        async def drain() -> None:
            stream = rtc.AudioStream.from_track(track=track)
            try:
                async for event in stream:
                    frame = event.frame
                    await sink.push(language, bytes(frame.data), frame.sample_rate)
            finally:
                await stream.aclose()

        task = asyncio.create_task(drain())
        drains.add(task)
        task.add_done_callback(drains.discard)

    await room.connect(grant["url"], grant["token"], rtc.RoomOptions(auto_subscribe=False))
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            want(publication)
    log.info("joined room for a language", extra={"language": language, "url": grant["url"]})

    await stop.wait()
    for task in list(drains):
        task.cancel()
    await room.disconnect()


async def status_forever(sink: Aes67Sink, languages: list[str], every: float = 10.0) -> None:
    while True:
        await asyncio.sleep(every)
        depths = {lang: round(sink.queue_depth(lang), 2) for lang in languages}
        log.info("aes67 agent", extra={"buffered_s": depths, **sink.stats.as_dict()})


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, help="join service base URL (the Cloud Run address)")
    ap.add_argument("--room", required=True)
    ap.add_argument("--profile", type=Path, required=True, help="device JSON from the console")
    ap.add_argument("--interface", default="0.0.0.0", help="IP of the NIC on the Dante network")
    ap.add_argument("--multicast", default="239.69.1.1", help="first flow's group")
    ap.add_argument("--port", type=int, default=5004)
    ap.add_argument("--ptp-grandmaster", default="00-00-00-00-00-00-00-00")
    ap.add_argument("--ptp-domain", type=int, default=0)
    ap.add_argument("--no-sap", action="store_true", help="do not announce over SAP")
    ap.add_argument("--insecure", action="store_true", help="accept a dev TLS certificate")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure(args.log_level)
    device = load_profile(args.profile)
    languages = sorted({c.language for c in device.channels if c.enabled})
    if not languages:
        log.error("the profile maps no languages; nothing to do", extra={"device": device.name})
        return 1
    if args.ptp_grandmaster == "00-00-00-00-00-00-00-00":
        log.warning(
            "no --ptp-grandmaster given; Dante receivers will show a clock warning "
            "until the host runs linuxptp and this names its grandmaster"
        )

    sink = Aes67Sink(
        device,
        Aes67Config(
            multicast_base=args.multicast,
            port=args.port,
            interface_ip=args.interface,
            ptp_grandmaster=args.ptp_grandmaster,
            ptp_domain=args.ptp_domain,
            session_name=device.name,
            sap=not args.no_sap,
        ),
    )
    ctx = _ssl_context(args.insecure)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    grants: dict[str, dict] = {}
    try:
        await sink.publish_languages(languages)
        for lang in languages:
            grants[lang] = join(args.base, args.room, lang, ctx)
        for flow in sink.flows:
            log.info("flow", extra=flow)

        tasks = [asyncio.create_task(listen(lang, grants[lang], sink, stop)) for lang in languages]
        tasks.append(asyncio.create_task(status_forever(sink, languages)))
        await stop.wait()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    finally:
        for grant in grants.values():
            leave(args.base, grant["listener_id"], grant["session_id"], ctx)
        await sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
