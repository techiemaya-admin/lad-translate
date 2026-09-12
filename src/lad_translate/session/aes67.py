"""
AES67 output: translated audio as RTP multicast on the venue's audio network.

The brief asked for a Dante Virtual Soundcard sink and then noted the problem
with it: DVS runs on Windows and macOS only, so the output host has to be a
laptop, and a laptop cannot host the pipeline. AES67 is the way round that.
Dante devices receive AES67 flows natively once "AES67 mode" is switched on in
Dante Controller, and an AES67 sender is nothing but UDP - it runs anywhere
with a network card, including a Raspberry Pi in the rack. So the output agent
is a Linux process, not a soundcard driver, and the IR transmitter's Dante
inputs subscribe to it like any other flow.

What a flow is, in the terms Dante Controller shows:

    RTP over UDP multicast, group 239.69.x.x, port 5004
    L24 big-endian PCM at 48 kHz, up to 8 channels interleaved
    one packet every millisecond: 48 frames, header + 48 * ch * 3 bytes
    announced over SAP (239.255.255.255:9875) with an SDP, every 30 seconds

A DEVICE HERE IS A SET OF FLOWS. config.OutputDevice models a card with N
channels; Dante accepts at most 8 channels per AES67 flow, so a 16-channel
device is up to two flows on two consecutive groups, channels 1-8 and 9-16.
The channel map's `channel` index is the position on the device, and this
module does the split. An operator patching "French on channel 10" does not
need to know that means channel 2 of flow 2. A flow is trimmed to its last
patched channel and a flow with nothing patched is not sent at all, because
an 8-channel L24 flow is 9.2 Mbit/s of multicast whether or not it carries
anything.

IT IS CLOCKED, AND THAT CHANGES EVERYTHING ABOUT ITS SHAPE. LiveKit's
AudioSource is pushed when there is speech and sends nothing between phrases.
An AES67 receiver expects a packet every millisecond whether or not anyone is
talking, and a gap is not silence to it - it is a dropped flow. So this is a
ring buffer per channel and a pump thread that sends on a 1 ms schedule
forever, filling with zeros when a ring is empty. push() writes into rings;
queue_depth() is how full they are. Nothing about a phrase's timing survives
into the packet stream, which is the point.

THE MEDIA CLOCK IS PTP, AND THIS PROCESS DOES NOT DO PTP. AES67 stamps RTP
timestamps from IEEE 1588 time - `a=mediaclk:direct=0` means "the 48 kHz
sample count since the PTP epoch". The sender here stamps from the system
clock. On a host running linuxptp (ptp4l + phc2sys) the system clock IS PTP
time and the stream locks; on a host without it the packets are valid RTP that
a software receiver will play, but a Dante device will show the flow with a
clock error and may refuse to subscribe. That is a deployment requirement,
stated in the SDP's ts-refclk line and in the log at start-up, not something
this code can paper over. See tools/output_agent.py for the host checklist.

Verified against a loopback receiver: framing, sequence and timestamp
continuity, channel interleave and gain, silence fill, rate. NOT yet verified
against a Dante device in AES67 mode or any PTP-locked receiver. The first
time it meets one, expect to adjust the SDP - Dante is particular about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import OutputChannel, OutputDevice
from ..obs.log import get_logger
from .pcm import Ring, resample, resampler_is_soxr

log = get_logger(__name__)

RATE = 48_000
"""AES67's mandatory rate. Dante runs cards at 48 kHz by default; a profile
saved at anything else is refused in publish_languages rather than resampled
to a rate the receiver will not accept."""

PTIME_MS = 1
FRAMES_PER_PACKET = RATE * PTIME_MS // 1000
"""48 frames. AES67 requires every receiver to accept 1 ms packets, and it is
what Dante devices send themselves, so it is the interoperable choice even
though 4 ms would be kinder to a Python pump."""

MAX_CHANNELS_PER_FLOW = 8
"""Dante's AES67 receive limit per flow."""

RING_S = 2.0
"""Per-channel buffer. The drift controller keeps playout within a few seconds
of live; two is enough to absorb a phrase arriving early and short enough that
a stalled consumer is noticed rather than buffered."""

PUSH_WAIT_TIMEOUT_S = 5.0
"""How long push() waits for ring space before dropping the phrase. Matches
session/room.py: a phrase that cannot be placed inside five seconds belongs
to a moment the audience has already left."""

SAP_GROUP = ("239.255.255.255", 9875)
SAP_INTERVAL_S = 30.0

RTP_VERSION = 2
L24_BYTES = 3


@dataclass(frozen=True)
class Aes67Config:
    multicast_base: str = "239.69.1.1"
    """First flow's group; each further flow takes the next address. Dante's
    own AES67 flows live in 239.69.0.0/16, which is why the default does."""

    port: int = 5004
    payload_type: int = 98
    """Dynamic. 98 is what Dante advertises for L24, and matching it costs
    nothing."""

    ttl: int = 16
    interface_ip: str = "0.0.0.0"
    """Which NIC multicast leaves on. A laptop with Wi-Fi and a cabled Dante
    port must send on the cabled one; the OS default is usually wrong."""

    ptp_grandmaster: str = "00-00-00-00-00-00-00-00"
    """Written into the SDP's ts-refclk line. Dante receivers compare it with
    the GM they follow; a mismatch is a visible clock warning, which is more
    honest than omitting the line."""

    ptp_domain: int = 0
    session_name: str = "LAD Live Translation"
    sap: bool = True
    sap_interval_s: float = SAP_INTERVAL_S


@dataclass
class SinkStats:
    packets: int = 0
    pushes: int = 0
    dropped_phrases: int = 0
    dropped_seconds: float = 0.0
    late_catchups: int = 0
    """Times the pump found itself more than 20 ms behind schedule and skipped
    ahead rather than bursting the missing packets. A receiver sees a gap
    either way; bursting also overflows its buffer."""

    def as_dict(self) -> dict:
        return {
            "packets": self.packets,
            "pushes": self.pushes,
            "dropped_phrases": self.dropped_phrases,
            "dropped_seconds": round(self.dropped_seconds, 2),
            "late_catchups": self.late_catchups,
        }


# --- audio ------------------------------------------------------------------

def to_48k(pcm: bytes, sample_rate: int) -> np.ndarray:
    """int16 mono PCM at any rate -> float32 at 48 kHz. See session/pcm.py."""
    return resample(pcm, sample_rate, RATE)


def pack_l24(frames: np.ndarray) -> bytes:
    """
    float32 (frames, channels) -> interleaved 24-bit big-endian PCM.

    Scale to int32 with 24 significant bits, view as big-endian bytes, and drop
    the top byte of each sample - which for a value that fits in 24 bits is
    sign extension and carries nothing.
    """
    clipped = np.clip(frames, -1.0, 1.0)
    as_int = (clipped * 8_388_607.0).astype(">i4")
    return as_int.reshape(-1).view(np.uint8).reshape(-1, 4)[:, 1:4].tobytes()


def unpack_l24(payload: bytes, channels: int) -> np.ndarray:
    """The inverse, for tests and receivers. (frames, channels) float32."""
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
    padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
    padded[:, 1:4] = raw
    as_int = padded.reshape(-1).view(">i4").astype(np.int32)
    # Sign-extend from 24 bits.
    as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
    return (as_int.astype(np.float32) / 8_388_607.0).reshape(-1, channels)


@dataclass
class _Flow:
    index: int
    group: str
    channels: list[OutputChannel]
    """Device channels carried, in flow-channel order. Gaps in the device's
    numbering are carried as silent channels so that a receiver's channel N
    is always the device's channel (flow_index * 8 + N)."""

    rings: list[Ring | None]
    """One per flow channel; None where the device has nothing mapped."""

    gains: np.ndarray
    ssrc: int = field(default_factory=lambda: random.getrandbits(32))
    seq: int = field(default_factory=lambda: random.getrandbits(16))
    sock: socket.socket | None = None

    @property
    def channel_count(self) -> int:
        return len(self.rings)


class Aes67Sink:
    """
    An AudioSink that emits the device's channel map as AES67 flows.

    Construct with the device profile the console saved. Nothing is opened
    until publish_languages(), which is where a session commits to a language
    list; a profile that maps a language the session does not publish gets a
    warning and a silent channel rather than an error, because the rig being
    patched for an event with four languages and running one with three is
    ordinary.
    """

    def __init__(self, device: OutputDevice, config: Aes67Config | None = None) -> None:
        self.device = device
        self.config = config or Aes67Config()
        self.stats = SinkStats()
        self._flows: list[_Flow] = []
        self._by_language: dict[str, list[tuple[Ring, float]]] = {}
        self._pump: threading.Thread | None = None
        self._running = threading.Event()
        self._sap_task: asyncio.Task | None = None
        self._sap_sock: socket.socket | None = None
        self._started_at = 0.0
        self._base_ts = 0

    # --- AudioSink -----------------------------------------------------------

    async def publish_languages(self, languages: list[str]) -> None:
        device = self.device
        if not device.enabled:
            raise RuntimeError(f"output device {device.name!r} is disabled in its profile")
        if device.sample_rate != RATE:
            raise RuntimeError(
                f"{device.name!r} is profiled at {device.sample_rate} Hz; AES67 flows are "
                f"{RATE} Hz. Change the card's rate in Dante Controller and the profile."
            )
        if self._flows:
            raise RuntimeError("publish_languages() was already called on this sink")

        mapped = {c.language for c in device.channels if c.enabled}
        for lang in sorted(mapped - set(languages)):
            log.warning(
                "channel map names a language this session does not publish; "
                "its channels will carry silence",
                extra={"language": lang, "device": device.name},
            )

        self._flows = self._build_flows()
        for flow in self._flows:
            flow.sock = self._open_socket()

        self._started_at = time.perf_counter()
        # AES67 media clock: 48 kHz sample count since the PTP epoch, which is
        # the system clock on a PTP-disciplined host. See the module docstring.
        self._base_ts = int(time.time() * RATE) & 0xFFFFFFFF
        self._running.set()
        self._pump = threading.Thread(target=self._run_pump, name="aes67-pump", daemon=True)
        self._pump.start()

        if self.config.sap:
            self._sap_sock = self._open_socket()
            self._sap_task = asyncio.create_task(self._announce_forever())

        log.info(
            "AES67 flows open",
            extra={
                "device": device.name,
                "flows": [
                    {"group": f.group, "port": self.config.port, "channels": f.channel_count}
                    for f in self._flows
                ],
                "languages": sorted(self._by_language),
                "silent_languages": sorted(mapped - set(languages)),
                "ptp_grandmaster": self.config.ptp_grandmaster,
                "resampler": "soxr" if resampler_is_soxr() else "linear",
            },
        )

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        targets = self._by_language.get(language)
        if not targets:
            # Not an error: the session publishes languages the rig does not
            # carry all the time. The phones still get it.
            return
        samples = to_48k(pcm, sample_rate)
        if samples.size == 0:
            return
        self.stats.pushes += 1

        # Wait for room rather than overwrite: a ring that is full means the
        # pump is behind the pipeline, and the drift controller upstream is
        # already reading queue_depth and speeding up or skipping. Overwriting
        # here would hide that from it.
        deadline = time.monotonic() + PUSH_WAIT_TIMEOUT_S
        while any(ring.free() < samples.size for ring, _ in targets):
            if time.monotonic() > deadline:
                self.stats.dropped_phrases += 1
                self.stats.dropped_seconds += samples.size / RATE
                log.error(
                    "AES67 ring full; phrase dropped",
                    extra={
                        "language": language,
                        "seconds": round(samples.size / RATE, 2),
                        "queue_depth_s": round(self.queue_depth(language), 2),
                    },
                )
                return
            await asyncio.sleep(0.005)
        for ring, gain in targets:
            ring.write(samples * gain)

    def queue_depth(self, language: str) -> float:
        targets = self._by_language.get(language)
        if not targets:
            return 0.0
        return max(ring.seconds() for ring, _ in targets)

    async def close(self) -> None:
        self._running.clear()
        if self._pump is not None:
            self._pump.join(timeout=1.0)
            self._pump = None
        if self._sap_task is not None:
            self._sap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sap_task
            self._sap_task = None
            # Tell receivers the flows are gone, so Dante Controller does not
            # list a dead session for the next twenty minutes.
            for flow in self._flows:
                self._send_sap(flow, deletion=True)
        if self._sap_sock is not None:
            self._sap_sock.close()
            self._sap_sock = None
        for flow in self._flows:
            if flow.sock is not None:
                flow.sock.close()
                flow.sock = None
        log.info("AES67 flows closed", extra={"device": self.device.name, **self.stats.as_dict()})

    # --- flows ---------------------------------------------------------------

    def _build_flows(self) -> list[_Flow]:
        device = self.device
        by_channel = {c.channel: c for c in device.channels if c.enabled}
        base = ipaddress.ip_address(self.config.multicast_base)
        flows: list[_Flow] = []
        count = device.channel_count
        for start in range(1, count + 1, MAX_CHANNELS_PER_FLOW):
            span = list(range(start, min(start + MAX_CHANNELS_PER_FLOW, count + 1)))
            # Trim to the last mapped channel and skip a flow with none. An
            # 8-channel L24 flow is 9.2 Mbit/s of multicast whether or not
            # anything is on it; a 16-channel card with four channels patched
            # should cost one 4-channel flow, not two of eight.
            mapped_positions = [i for i, ch in enumerate(span) if ch in by_channel]
            if not mapped_positions:
                continue
            span = span[: mapped_positions[-1] + 1]

            rings: list[Ring | None] = []
            gains = np.zeros(len(span), dtype=np.float32)
            carried: list[OutputChannel] = []
            for position, channel_number in enumerate(span):
                channel = by_channel.get(channel_number)
                if channel is None:
                    rings.append(None)
                    continue
                ring = Ring(int(RING_S * RATE), RATE)
                gain = float(10 ** (channel.gain_db / 20.0))
                rings.append(ring)
                gains[position] = gain
                carried.append(channel)
                self._by_language.setdefault(channel.language, []).append((ring, gain))
            index = len(flows)
            flows.append(
                _Flow(index=index, group=str(base + index), channels=carried, rings=rings, gains=gains)
            )
        return flows

    def _open_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.config.ttl)
        if self.config.interface_ip != "0.0.0.0":
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(self.config.interface_ip),
            )
        return sock

    # --- the pump ------------------------------------------------------------

    def _run_pump(self) -> None:
        """
        One packet per flow per millisecond, on a schedule, forever.

        The schedule is absolute - packet n goes at start + n ms - so jitter
        in one iteration does not accumulate into the next. If the thread
        wakes more than 20 ms late (the machine stalled), it skips ahead
        rather than sending the backlog in a burst: the receiver has already
        played silence for that gap, and a burst on top would overflow its
        buffer and cost a second gap.
        """
        period = PTIME_MS / 1000.0
        frames = np.zeros(FRAMES_PER_PACKET, dtype=np.float32)
        n = 0
        while self._running.is_set():
            target = self._started_at + n * period
            now = time.perf_counter()
            if now - target > 0.020:
                skipped = int((now - target) / period)
                n += skipped
                self.stats.late_catchups += 1
                continue
            if target > now:
                remaining = target - now
                if remaining > 0.0003:
                    time.sleep(remaining - 0.0002)
                while time.perf_counter() < target:
                    pass
            ts = (self._base_ts + n * FRAMES_PER_PACKET) & 0xFFFFFFFF
            for flow in self._flows:
                self._send_packet(flow, ts, frames)
            n += 1

    def _send_packet(self, flow: _Flow, ts: int, scratch: np.ndarray) -> None:
        block = np.zeros((FRAMES_PER_PACKET, flow.channel_count), dtype=np.float32)
        for position, ring in enumerate(flow.rings):
            if ring is None:
                continue
            ring.read(FRAMES_PER_PACKET, scratch)
            block[:, position] = scratch
        header = struct.pack(
            "!BBHII",
            RTP_VERSION << 6,
            self.config.payload_type & 0x7F,
            flow.seq,
            ts,
            flow.ssrc,
        )
        flow.seq = (flow.seq + 1) & 0xFFFF
        try:
            flow.sock.sendto(header + pack_l24(block), (flow.group, self.config.port))
            self.stats.packets += 1
        except OSError as exc:
            # Logged, not raised: this thread has no caller. FanOutSink sees
            # the failure through queue_depth/push if the socket is dead, and
            # a transient send error on a loaded NIC is not worth a gap.
            log.warning("AES67 send failed", extra={"group": flow.group, "error": str(exc)})

    # --- SAP / SDP ---------------------------------------------------------

    def sdp(self, flow: _Flow) -> str:
        """
        The session description a receiver needs to subscribe.

        Lines a Dante device looks for: L24/48000/<ch>, ptime:1, ts-refclk
        with a PTP grandmaster, mediaclk:direct=0. The origin address is the
        sending interface where it is known; 0.0.0.0 means "the OS chose".
        """
        cfg = self.config
        origin = cfg.interface_ip if cfg.interface_ip != "0.0.0.0" else _local_ip()
        session_id = self._base_ts or 1
        name = cfg.session_name if len(self._flows) == 1 else f"{cfg.session_name} {flow.index + 1}"
        return "\r\n".join(
            [
                "v=0",
                f"o=- {session_id} {session_id} IN IP4 {origin}",
                f"s={name}",
                f"c=IN IP4 {flow.group}/{cfg.ttl}",
                "t=0 0",
                f"a=clock-domain:PTPv2 {cfg.ptp_domain}",
                f"m=audio {cfg.port} RTP/AVP {cfg.payload_type}",
                f"a=rtpmap:{cfg.payload_type} L24/{RATE}/{flow.channel_count}",
                "a=recvonly",
                f"a=ptime:{PTIME_MS}",
                f"a=framecount:{FRAMES_PER_PACKET}",
                f"a=ts-refclk:ptp=IEEE1588-2008:{cfg.ptp_grandmaster}:{cfg.ptp_domain}",
                "a=mediaclk:direct=0",
                "",
            ]
        )

    def sap_packet(self, flow: _Flow, deletion: bool = False) -> bytes:
        """RFC 2974. Version 1, IPv4 origin, no auth, payload type stated."""
        cfg = self.config
        origin = cfg.interface_ip if cfg.interface_ip != "0.0.0.0" else _local_ip()
        flags = 0x20 | (0x04 if deletion else 0x00)
        msg_id = (flow.ssrc + flow.index) & 0xFFFF
        header = struct.pack("!BBH", flags, 0, msg_id) + socket.inet_aton(origin)
        return header + b"application/sdp\x00" + self.sdp(flow).encode()

    def _send_sap(self, flow: _Flow, deletion: bool = False) -> None:
        if self._sap_sock is None:
            return
        try:
            self._sap_sock.sendto(self.sap_packet(flow, deletion), SAP_GROUP)
        except OSError as exc:
            log.warning("SAP announce failed", extra={"group": flow.group, "error": str(exc)})

    async def _announce_forever(self) -> None:
        while True:
            for flow in self._flows:
                self._send_sap(flow)
            await asyncio.sleep(self.config.sap_interval_s)

    # --- introspection ------------------------------------------------------

    @property
    def flows(self) -> list[dict]:
        """What is on the wire, for the agent's status line and for tests."""
        return [
            {
                "group": f.group,
                "port": self.config.port,
                "channels": f.channel_count,
                "carried": [
                    {"channel": c.channel, "language": c.language, "ir_channel": c.ir_channel}
                    for c in f.channels
                ],
            }
            for f in self._flows
        ]


def _local_ip() -> str:
    """Best guess at the address other hosts see this one as; for SDP only."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("239.69.1.1", 5004))
            return probe.getsockname()[0]
    except OSError:
        return os.getenv("LAD_AES67_ORIGIN_IP", "127.0.0.1")
