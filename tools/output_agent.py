#!/usr/bin/env python3
"""
The venue's output agent: translated audio from the room, out to the rig.

Runs on a machine at the venue and does what a phone does, N times over:
joins the room as a listener for each language the channel map names, and
instead of playing what it hears, hands it to one of two engines. Which one
is the profile's Kind:

    aes67                          RTP multicast on the Dante/AES67 network
                                   (session/aes67.py). Any Linux box on the
                                   VLAN; needs linuxptp; received by Dante
                                   HARDWARE with AES67 mode on.
    dante-vsc / coreaudio / asio / alsa
                                   a sound card THIS machine can see
                                   (session/localcard.py). Dante Virtual
                                   Soundcard on a Mac or PC is the usual one:
                                   the agent plays each language into DVS's
                                   channels and DVS puts them on the Dante
                                   network as a transmitter.

Dante Virtual Soundcard does not receive AES67, so a venue running DVS wants
the second. A rack box with no sound card wants the first.

It sits at the venue because both engines need the venue's audio network and
the pipeline is in a cloud region. It reuses the listener path because the
listener path is the one that is proven: if a phone can hear a language, so
can this.

    # a Mac with Dante Virtual Soundcard, profile Kind dante-vsc
    python tools/output_agent.py --base https://lad-translate-dev-...run.app \\
        --room dubai-demo --profile mac-avc.json

    # a Linux box on the Dante VLAN, profile Kind aes67
    python tools/output_agent.py --base https://lad-translate-dev-...run.app \\
        --room dubai-demo --profile main-hall.json \\
        --interface 192.168.10.5 --ptp-grandmaster 00-1d-c1-ff-fe-12-34-56

    # what this machine can play to, for the profile's "Host audio device"
    python tools/output_agent.py --list-devices

The profile is the device as the console saved it: "Profile JSON" on the
device's card, or GET /console/api/outputs/devices/<id>.

HOST CHECKLIST for a sound card (dante-vsc and friends):

  1. DVS is installed, licensed and running, and Dante Controller shows this
     machine as a device on the network. --list-devices must show it.
  2. The profile's sample rate matches DVS's (Dante Controller > Device Config
     > Sample Rate; 48000 unless someone changed it). A mismatch is refused at
     open rather than played at the wrong pitch.
  3. In Dante Controller, subscribe the IR transmitter's inputs to this
     machine's transmit channels 1..N - the same numbers as the channel map.
  4. Nothing else has the device open exclusively (some DAWs do).

HOST CHECKLIST for AES67, in the order things go wrong:

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
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.config import OutputChannel, OutputDevice
from lad_translate.obs.log import configure, get_logger
from lad_translate.session.aes67 import Aes67Config, Aes67Sink
from lad_translate.session.localcard import LocalCardConfig, LocalCardSink, list_output_devices

CARD_KINDS = ("dante-vsc", "coreaudio", "asio", "alsa")

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


class NoLiveSession(Exception):
    """The room exists as a name but nothing is running in it right now."""


class SingleDrain:
    """
    One running task, replaced rather than added to.

    A session restart republishes every language track, the room
    re-subscribes, and the subscribe handler fires again. Starting another
    drain there leaves the previous one running into the same ring, so the
    sink receives N copies of every phrase. Five copies, staggered by five
    restarts, is audio that does not stop when the speaker does - measured
    on a Mac feeding Dante Virtual Soundcard, where the relay channel also
    sat permanently two seconds behind, because two streams were filling a
    buffer that drains at one.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.replaced = 0

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def replace(self, coro) -> asyncio.Task:
        """Cancel whatever is running and start this instead."""
        if self.running:
            self.replaced += 1
            self.task.cancel()
        self.task = asyncio.ensure_future(coro)
        return self.task

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
            self.task = None


def join(base: str, room: str, language: str, ctx: ssl.SSLContext) -> dict:
    """Ask the join API for a listener token, exactly as the browser page does."""
    req = urllib.request.Request(
        f"{base}/api/rooms/{room}/join",
        data=json.dumps({"language": language}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # "no live session in room" - the session ended, or has not been
            # started yet. Either is ordinary at a venue: the agent is often
            # switched on before the operator presses APPLY.
            raise NoLiveSession(room) from exc
        raise


async def wait_for_room(base: str, room: str, ctx: ssl.SSLContext, stop: asyncio.Event) -> bool:
    """
    Block until the room has a live session, or stop is set.

    Polls every few seconds and logs once a minute, so the operator sees
    "waiting for dubai-demo" rather than a stack trace, and the agent is
    already attached the moment the session comes up.
    """
    attempts = 0
    while not stop.is_set():
        try:
            req = urllib.request.Request(f"{base}/api/rooms/{room}")
            with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
                info = json.load(response)
            if info.get("status") in ("starting", "live"):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            log.warning("join service unreachable; retrying", extra={"error": str(exc)[:120]})
        if attempts % 12 == 0:
            log.info(
                "waiting for a live session in the room",
                extra={"room": room, "hint": "start it from the console's deck (APPLY)"},
            )
        attempts += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=5.0)
        except TimeoutError:
            continue
    return False


def leave(base: str, listener_id: str, session_id: str, ctx: ssl.SSLContext) -> None:
    req = urllib.request.Request(
        f"{base}/api/listeners/{listener_id}/leave?session_id={session_id}",
        data=b"",
        method="POST",
    )
    with contextlib.suppress(Exception):
        urllib.request.urlopen(req, context=ctx, timeout=10).close()


async def listen(language: str, grant: dict, sink, stop: asyncio.Event) -> None:
    """One room connection, one language track, straight into the sink."""
    import livekit.rtc as rtc

    room = rtc.Room()
    want_name = grant["track_name"]
    drain = SingleDrain()   # one stream per language; see the class

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
        log.info(
            "language track re-attached; dropping the previous stream" if drain.running
            else "language track attached",
            extra={"language": language, "track": want_name},
        )

        async def pump() -> None:
            stream = rtc.AudioStream.from_track(track=track)
            try:
                async for event in stream:
                    frame = event.frame
                    await sink.push(language, bytes(frame.data), frame.sample_rate)
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()

        drain.replace(pump())

    await room.connect(grant["url"], grant["token"], rtc.RoomOptions(auto_subscribe=False))
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            want(publication)
    log.info("joined room for a language", extra={"language": language, "url": grant["url"]})

    await stop.wait()
    await drain.stop()
    await room.disconnect()


async def status_forever(sink, languages: list[str], every: float = 10.0) -> None:
    while True:
        await asyncio.sleep(every)
        depths = {lang: round(sink.queue_depth(lang), 2) for lang in languages}
        log.info("output agent", extra={"buffered_s": depths, **sink.stats.as_dict()})


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", help="join service base URL (the Cloud Run address)")
    ap.add_argument("--room")
    ap.add_argument("--profile", type=Path, help="device JSON from the console")
    ap.add_argument("--engine", choices=("auto", "aes67", "card"), default="auto",
                    help="auto picks from the profile's kind")
    ap.add_argument("--device", help="override the profile's host audio device (card engine)")
    ap.add_argument("--list-devices", action="store_true",
                    help="print this machine's audio output devices and exit")
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

    if args.list_devices:
        devices = list_output_devices()
        if not devices:
            print("  no audio output devices on this machine")
            return 1
        print("  this machine can play to:")
        for d in devices:
            print(f"    {d.name!r:44s} {d.max_output_channels:3d} ch  {int(d.default_samplerate)} Hz  ({d.hostapi})")
        print('  put the name in the profile\'s "Host audio device" - matching is forgiving.')
        return 0

    if not (args.base and args.room and args.profile):
        ap.error("--base, --room and --profile are required (or --list-devices)")

    device = load_profile(args.profile)
    languages = sorted({c.language for c in device.channels if c.enabled})
    if not languages:
        log.error("the profile maps no languages; nothing to do", extra={"device": device.name})
        return 1
    if not device.enabled:
        log.error(
            "the profile is disabled; tick 'Device enabled' in the console and export it again",
            extra={"device": device.name},
        )
        return 1

    engine = args.engine
    if engine == "auto":
        engine = "aes67" if device.kind == "aes67" else "card"
    if engine == "aes67" and device.kind in CARD_KINDS:
        log.warning("running AES67 for a profile whose kind is a local card", extra={"kind": device.kind})
    if engine == "card" and device.kind == "aes67":
        log.warning("running the card engine for a profile whose kind is aes67", extra={"kind": device.kind})

    if engine == "aes67":
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
    else:
        sink = LocalCardSink(device, LocalCardConfig(device_name=args.device))
    ctx = _ssl_context(args.insecure)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    grants: dict[str, dict] = {}
    try:
        await sink.publish_languages(languages)
        # The card is open and playing silence from here; the room may take
        # a while. A session that ends mid-run is handled the same way: the
        # listeners drop, and we go back to waiting rather than exiting.
        if not await wait_for_room(args.base, args.room, ctx, stop):
            return 0
        for lang in languages:
            try:
                grants[lang] = join(args.base, args.room, lang, ctx)
            except NoLiveSession:
                log.error("the session ended while joining; start it again and rerun", extra={"room": args.room})
                return 1
        if engine == "aes67":
            for flow in sink.flows:
                log.info("flow", extra=flow)
        else:
            log.info("card", extra={"device": sink.card.name, "patch": sink.patch})

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
