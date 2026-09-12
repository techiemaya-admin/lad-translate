"""
Local card output: translated audio played into a sound device on this machine.

The device is whatever the operating system calls it - "Dante Virtual
Soundcard" on a Mac or PC with DVS installed, a Dante PCIe card, a USB
interface, or the built-in output for a check. PortAudio opens it, which is
why one sink covers four of the kinds in the console's dropdown: dante-vsc,
coreaudio, asio and alsa are all "a card this machine can see".

WHY THIS EXISTS NEXT TO AES67. Dante Virtual Soundcard receives Dante flows
only; per Audinate it does not support AES67 mode, so the AES67 agent's
flows never appear in it. A venue running DVS on a Mac needs the audio
played INTO DVS as a sound card, and DVS then puts it on the Dante network
like any other transmitter. That is this sink. The trade against AES67: no
PTP to configure and no multicast to route, but the output host must be a
Mac or PC with DVS licensed, on the Dante network.

IT IS CLOCKED BY THE CARD. PortAudio calls back every few milliseconds
asking for the next block of frames for every channel, and the callback has
to answer inside that window or the card plays a gap. So this is the same
shape as the AES67 sink - a ring per mapped channel, silence when a ring is
empty - with the card's callback in place of the pump thread. The callback
does nothing but copy: no allocation, no logging, no locks held longer than
a memcpy. Everything slow happens in push(), on the event loop.

THE CHANNEL MAP IS THE DEVICE'S. Channel 3 in the console is channel 3 on the
card. The stream is opened with as many channels as the highest patched one
(a 64-channel DVS with French on 2 and Arabic on 3 opens 3 channels), and the
card leaves the rest silent. The profile's sample rate is what the stream
asks the card for; Piper's 22050 is resampled to it in push(). DVS is
normally 48000, set in Dante Controller; a mismatch is refused at open with
the rate the card reports, rather than played at the wrong pitch.

Verified on this machine against Dante Virtual Soundcard 64x64 at 48 kHz and
the built-in output: the stream opens by name, the callback keeps up with no
underruns, and a tone pushed to a language comes out of exactly its channel.
NOT verified: a Dante receiver subscribed to DVS's transmit channels - that
needs the Dante network and Dante Controller. See tools/output_agent.py.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import OutputChannel, OutputDevice
from ..obs.log import get_logger
from .pcm import Ring, resample, resampler_is_soxr

log = get_logger(__name__)

RING_S = 2.0
PUSH_WAIT_TIMEOUT_S = 5.0
"""Same numbers and the same reasons as session/aes67.py."""

BLOCK_FRAMES = 480
"""10 ms at 48 kHz per callback. Small enough that a phrase reaches the
card within a frame of arriving, large enough that Python keeps up: a 10 ms
callback doing a few array copies runs well under 1 ms."""


@dataclass(frozen=True)
class LocalCardConfig:
    device_name: str | None = None
    """Overrides the profile's device_name. The console's placeholder is
    "Dante Virtual Soundcard" and someone will type "Dante Virtual Sound
    card"; matching is forgiving, see find_device()."""

    block_frames: int = BLOCK_FRAMES
    latency: str | float = "low"
    """PortAudio's suggested latency. "low" asks the host API for its
    smallest safe buffer, which for CoreAudio and DVS is a few milliseconds."""


@dataclass
class CardStats:
    callbacks: int = 0
    frames_out: int = 0
    underruns: int = 0
    """Callbacks PortAudio flagged as late (output underflow). One is a
    click; a run of them is a machine that cannot keep up."""

    pushes: int = 0
    dropped_phrases: int = 0
    dropped_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "callbacks": self.callbacks,
            "frames_out": self.frames_out,
            "underruns": self.underruns,
            "pushes": self.pushes,
            "dropped_phrases": self.dropped_phrases,
            "dropped_seconds": round(self.dropped_seconds, 2),
        }


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_output_channels: int
    default_samplerate: float
    hostapi: str

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "max_output_channels": self.max_output_channels,
            "default_samplerate": self.default_samplerate,
            "hostapi": self.hostapi,
        }


def _fold(name: str) -> str:
    """'Dante Virtual Sound card' == 'dante virtual soundcard' == 'DANTE-VIRTUAL-SOUNDCARD'."""
    return re.sub(r"[\s_\-]+", "", name).lower()


def list_output_devices() -> list[DeviceInfo]:
    """Every device this machine can play to, as PortAudio sees them."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    out = []
    for index, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            out.append(
                DeviceInfo(
                    index=index,
                    name=d["name"],
                    max_output_channels=int(d["max_output_channels"]),
                    default_samplerate=float(d["default_samplerate"]),
                    hostapi=apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "?",
                )
            )
    return out


def find_device(name: str, devices: list[DeviceInfo] | None = None) -> DeviceInfo:
    """
    The output device called `name`, forgivingly.

    Exact first; then ignoring case, spaces and dashes; then a unique
    substring. "Dante Virtual Sound card" finds "Dante Virtual Soundcard";
    "dante" finds it too if nothing else on the machine says Dante. Two
    matches is an error rather than a guess: the wrong card at a venue is
    the wrong room hearing French.
    """
    devices = list_output_devices() if devices is None else devices
    if not devices:
        raise LookupError("this machine has no audio output devices")
    for d in devices:
        if d.name == name:
            return d
    folded = _fold(name)
    same = [d for d in devices if _fold(d.name) == folded]
    if len(same) == 1:
        return same[0]
    contains = [d for d in devices if folded and folded in _fold(d.name)]
    if len(contains) == 1:
        return contains[0]
    names = ", ".join(repr(d.name) for d in devices)
    if len(contains) > 1:
        raise LookupError(f"{name!r} matches more than one output device: {names}")
    raise LookupError(f"no output device called {name!r}; this machine has: {names}")


class LocalCardSink:
    """
    An AudioSink that plays the device's channel map into a local sound card.

    `open_stream` exists for tests: it is called with (device_index, rate,
    channels, callback) and must return something with start(), stop() and
    close(). The default opens a PortAudio OutputStream.
    """

    def __init__(
        self,
        device: OutputDevice,
        config: LocalCardConfig | None = None,
        open_stream: Callable[..., Any] | None = None,
        devices: list[DeviceInfo] | None = None,
    ) -> None:
        self.device = device
        self.config = config or LocalCardConfig()
        self.stats = CardStats()
        self._open_stream = open_stream or self._open_portaudio
        self._devices = devices
        self._stream: Any = None
        self._card: DeviceInfo | None = None
        self._rings: list[Ring | None] = []
        """Indexed by stream channel (0-based = device channel - 1)."""

        self._by_language: dict[str, list[tuple[Ring, float]]] = {}
        self._channels: list[OutputChannel] = []
        self._scratch: np.ndarray = np.zeros(0, dtype=np.float32)
        self._started_at = 0.0

    # --- AudioSink -----------------------------------------------------------

    async def publish_languages(self, languages: list[str]) -> None:
        device = self.device
        if not device.enabled:
            raise RuntimeError(f"output device {device.name!r} is disabled in its profile")
        if self._stream is not None:
            raise RuntimeError("publish_languages() was already called on this sink")

        name = self.config.device_name or device.device_name
        card = find_device(name, self._devices)
        mapped = [c for c in device.channels if c.enabled]
        highest = max((c.channel for c in mapped), default=0)
        if highest == 0:
            raise RuntimeError(f"{device.name!r} has no enabled channels to play")
        if highest > card.max_output_channels:
            raise RuntimeError(
                f"{device.name!r} patches channel {highest} but {card.name!r} has "
                f"{card.max_output_channels} output channels"
            )

        for lang in sorted({c.language for c in mapped} - set(languages)):
            log.warning(
                "channel map names a language this session does not publish; "
                "its channels will carry silence",
                extra={"language": lang, "device": device.name},
            )

        rate = device.sample_rate
        self._rings = [None] * highest
        for c in mapped:
            ring = Ring(int(RING_S * rate), rate)
            self._rings[c.channel - 1] = ring
            gain = float(10 ** (c.gain_db / 20.0))
            self._by_language.setdefault(c.language, []).append((ring, gain))
        self._channels = mapped
        self._scratch = np.zeros(self.config.block_frames, dtype=np.float32)
        self._card = card

        try:
            self._stream = self._open_stream(card.index, rate, highest, self._callback)
            self._stream.start()
        except Exception as exc:
            # PortAudio's own message names the rate or the channel count it
            # refused; keep it, because "could not open device" is what an
            # operator reads at a venue and it says nothing.
            raise RuntimeError(f"could not open {card.name!r} at {rate} Hz x {highest} ch: {exc}") from exc

        self._started_at = time.monotonic()
        log.info(
            "local card open",
            extra={
                "device": device.name,
                "card": card.name,
                "hostapi": card.hostapi,
                "rate": rate,
                "channels": highest,
                "block_frames": self.config.block_frames,
                "languages": sorted(self._by_language),
                "silent_languages": sorted({c.language for c in mapped} - set(languages)),
                "resampler": "soxr" if resampler_is_soxr() else "linear",
            },
        )

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        targets = self._by_language.get(language)
        if not targets:
            return
        samples = resample(pcm, sample_rate, self.device.sample_rate)
        if samples.size == 0:
            return
        self.stats.pushes += 1
        deadline = time.monotonic() + PUSH_WAIT_TIMEOUT_S
        while any(ring.free() < samples.size for ring, _ in targets):
            if time.monotonic() > deadline:
                self.stats.dropped_phrases += 1
                self.stats.dropped_seconds += samples.size / self.device.sample_rate
                log.error(
                    "card ring full; phrase dropped",
                    extra={"language": language, "seconds": round(samples.size / self.device.sample_rate, 2)},
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
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001 - closing is best effort
                log.warning("card stream did not close cleanly", extra={"error": str(exc)})
            self._stream = None
        log.info("local card closed", extra={"device": self.device.name, **self.stats.as_dict()})

    # --- the card's callback ---------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """
        Fill (frames, channels) for the card. Runs on PortAudio's thread.

        Nothing here may block, allocate much, or log: a callback that is
        late is a gap the audience hears. The scratch buffer is reused; a
        block size different from the one configured (some hosts do that on
        the first callback) is handled by slicing, not by resizing.
        """
        self.stats.callbacks += 1
        self.stats.frames_out += frames
        # sounddevice's CallbackFlags is truthy when any flag is set;
        # output_underflow is the one that means we were late.
        if status and getattr(status, "output_underflow", False):
            self.stats.underruns += 1
        scratch = self._scratch if frames <= self._scratch.size else np.zeros(frames, dtype=np.float32)
        for index, ring in enumerate(self._rings):
            if ring is None:
                outdata[:, index] = 0.0
                continue
            ring.read(frames, scratch)
            outdata[:, index] = scratch[:frames]

    # --- PortAudio ------------------------------------------------------------

    def _open_portaudio(self, index: int, rate: int, channels: int, callback: Callable) -> Any:
        import sounddevice as sd

        return sd.OutputStream(
            device=index,
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=self.config.block_frames,
            latency=self.config.latency,
            callback=callback,
        )

    # --- introspection --------------------------------------------------------

    @property
    def card(self) -> DeviceInfo | None:
        return self._card

    @property
    def patch(self) -> list[dict]:
        return [
            {"channel": c.channel, "language": c.language, "ir_channel": c.ir_channel, "gain_db": c.gain_db}
            for c in self._channels
        ]


__all__ = [
    "CardStats",
    "DeviceInfo",
    "LocalCardConfig",
    "LocalCardSink",
    "find_device",
    "list_output_devices",
]
