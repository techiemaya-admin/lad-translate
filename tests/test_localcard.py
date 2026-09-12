"""
The local card sink, judged by what the card's callback is handed.

A fake stream stands in for PortAudio: it holds the callback and lets the
test pull blocks from it, exactly as a card would every ten milliseconds.
That covers the arithmetic - which channel gets which language at what
gain, silence where nothing is mapped, the ring emptying at the card's
rate - on a CI box with no audio hardware at all.

What a fake cannot tell you is whether PortAudio opens the real device
by name and keeps up. That is one test at the bottom, skipped unless
sounddevice imports and an output device exists, and it was run against a
real Dante Virtual Soundcard on the machine this was written on.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lad_translate.config import OutputChannel, OutputDevice
from lad_translate.session.localcard import (
    DeviceInfo,
    LocalCardConfig,
    LocalCardSink,
    find_device,
)
from lad_translate.session.sinks import AudioSink

RATE = 48_000
CARDS = [
    DeviceInfo(0, "Dante Virtual Soundcard", 64, 48000.0, "Core Audio"),
    DeviceInfo(2, "MacBook Pro Speakers", 2, 48000.0, "Core Audio"),
    DeviceInfo(5, "ZoomAudioDevice", 2, 48000.0, "Core Audio"),
]


def a_device(channels, **over) -> OutputDevice:
    fields = {
        "device_id": "d1", "name": "Mac - AVC", "device_name": "Dante Virtual Sound card",
        "kind": "dante-vsc", "channel_count": 64, "sample_rate": RATE, "channels": channels,
    }
    fields.update(over)
    return OutputDevice(**fields)


def tone(seconds: float, hz: float = 440.0, rate: int = 22050, amp: float = 0.5) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype(np.int16).tobytes()


class FakeStream:
    """What PortAudio would do, on demand instead of on a timer."""

    def __init__(self, index, rate, channels, callback) -> None:
        self.index, self.rate, self.channels, self.callback = index, rate, channels, callback
        self.started = self.closed = False

    def start(self) -> None: self.started = True
    def stop(self) -> None: self.started = False
    def close(self) -> None: self.closed = True

    def pull(self, frames: int = 480, underflow: bool = False) -> np.ndarray:
        out = np.zeros((frames, self.channels), dtype=np.float32)
        status = type("Flags", (), {"output_underflow": underflow, "__bool__": lambda s: underflow})()
        self.callback(out, frames, None, status)
        return out


def build(channels, languages=None, **over):
    opened = {}

    def open_stream(index, rate, chans, callback):
        opened["stream"] = FakeStream(index, rate, chans, callback)
        return opened["stream"]

    sink = LocalCardSink(a_device(channels, **over), open_stream=open_stream, devices=CARDS)
    return sink, opened


# --- finding the card ------------------------------------------------------------


def test_the_card_is_found_however_the_operator_typed_it():
    assert find_device("Dante Virtual Soundcard", CARDS).index == 0
    assert find_device("Dante Virtual Sound card", CARDS).index == 0
    assert find_device("dante-virtual-soundcard", CARDS).index == 0
    assert find_device("DANTE", CARDS).index == 0


def test_an_ambiguous_name_is_refused_not_guessed():
    two_dantes = [*CARDS, DeviceInfo(7, "Dante Via", 16, 48000.0, "Core Audio")]
    with pytest.raises(LookupError, match="more than one"):
        find_device("dante", two_dantes)
    with pytest.raises(LookupError, match="no output device"):
        find_device("RedNet", CARDS)


def test_the_error_lists_what_the_machine_has():
    with pytest.raises(LookupError, match="MacBook Pro Speakers"):
        find_device("RME", CARDS)


# --- the shape of the sink ----------------------------------------------------------


def test_it_is_an_audio_sink():
    sink, _ = build((OutputChannel("fr", 1),))
    assert isinstance(sink, AudioSink)


@pytest.mark.asyncio
async def test_the_stream_is_as_wide_as_the_highest_patched_channel():
    sink, opened = build((OutputChannel("fr", 2), OutputChannel("ar", 3)))
    await sink.publish_languages(["fr", "ar"])
    assert opened["stream"].channels == 3
    assert opened["stream"].rate == RATE
    assert opened["stream"].index == 0
    assert opened["stream"].started
    await sink.close()
    assert opened["stream"].closed


@pytest.mark.asyncio
async def test_a_channel_beyond_the_card_is_refused_at_open():
    sink, _ = build((OutputChannel("fr", 3),), device_name="MacBook Pro Speakers")
    with pytest.raises(RuntimeError, match="2 output channels"):
        await sink.publish_languages(["fr"])


@pytest.mark.asyncio
async def test_a_disabled_profile_is_refused_at_open():
    sink, _ = build((OutputChannel("fr", 1),), enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        await sink.publish_languages(["fr"])


@pytest.mark.asyncio
async def test_a_profile_with_nothing_enabled_is_refused():
    sink, _ = build((OutputChannel("fr", 1, enabled=False),))
    with pytest.raises(RuntimeError, match="no enabled channels"):
        await sink.publish_languages(["fr"])


@pytest.mark.asyncio
async def test_portaudios_refusal_is_kept_in_the_message():
    def refuse(index, rate, chans, callback):
        raise ValueError("Invalid sample rate")

    sink = LocalCardSink(a_device((OutputChannel("fr", 1),)), open_stream=refuse, devices=CARDS)
    with pytest.raises(RuntimeError, match="48000 Hz x 1 ch: Invalid sample rate"):
        await sink.publish_languages(["fr"])


# --- what the card is handed ----------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_before_anyone_speaks_and_on_unmapped_channels():
    sink, opened = build((OutputChannel("fr", 1), OutputChannel("ar", 3)))
    await sink.publish_languages(["fr", "ar"])
    block = opened["stream"].pull()
    assert block.shape == (480, 3)
    assert not block.any()
    await sink.close()


@pytest.mark.asyncio
async def test_speech_lands_on_its_own_channel_at_its_own_gain():
    """French on 1 at 0 dB and on 3 at -6 dB; Arabic on 2. Push French only."""
    sink, opened = build((
        OutputChannel("fr", 1),
        OutputChannel("ar", 2),
        OutputChannel("fr", 3, gain_db=-6.0),
    ))
    await sink.publish_languages(["fr", "ar"])
    await sink.push("fr", tone(0.3, hz=1000.0), 22050)   # Piper's rate -> 48k
    blocks = [opened["stream"].pull() for _ in range(25)]  # 250 ms
    audio = np.concatenate(blocks)
    rms = np.sqrt(np.mean(audio**2, axis=0))
    assert rms[0] > 0.05, "French did not reach channel 1"
    assert rms[1] < 1e-6, "Arabic's channel carried audio nobody pushed"
    assert abs(rms[2] / rms[0] - 10 ** (-6 / 20)) < 0.05, "channel 3 is not 6 dB down"
    await sink.close()


@pytest.mark.asyncio
async def test_the_ring_drains_at_the_cards_rate():
    sink, opened = build((OutputChannel("fr", 1),))
    await sink.publish_languages(["fr"])
    await sink.push("fr", tone(1.0), 22050)
    assert abs(sink.queue_depth("fr") - 1.0) < 0.01
    for _ in range(50):          # 50 x 10 ms = half a second of card time
        opened["stream"].pull()
    assert abs(sink.queue_depth("fr") - 0.5) < 0.01
    await sink.close()


@pytest.mark.asyncio
async def test_a_bigger_block_than_configured_is_handled_not_crashed():
    """Some host APIs hand a different block size on the first callback."""
    sink, opened = build((OutputChannel("fr", 1),))
    await sink.publish_languages(["fr"])
    await sink.push("fr", tone(0.1), 22050)
    block = opened["stream"].pull(frames=2048)
    assert block.shape == (2048, 1) and block.any()
    await sink.close()


@pytest.mark.asyncio
async def test_underruns_are_counted():
    sink, opened = build((OutputChannel("fr", 1),))
    await sink.publish_languages(["fr"])
    opened["stream"].pull(underflow=True)
    opened["stream"].pull(underflow=False)
    assert sink.stats.underruns == 1 and sink.stats.callbacks == 2
    await sink.close()


@pytest.mark.asyncio
async def test_a_language_the_card_does_not_carry_is_ignored():
    sink, _ = build((OutputChannel("fr", 1),))
    await sink.publish_languages(["fr", "de"])
    await sink.push("de", tone(0.1), 22050)
    assert sink.queue_depth("de") == 0.0 and sink.stats.pushes == 0
    await sink.close()


@pytest.mark.asyncio
async def test_a_full_ring_waits_then_drops_loudly(monkeypatch, caplog):
    import logging

    from lad_translate.session import localcard

    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(localcard, "PUSH_WAIT_TIMEOUT_S", 0.2)
    sink, _ = build((OutputChannel("fr", 1),))
    await sink.publish_languages(["fr"])
    await sink.push("fr", tone(1.9), 22050)
    await sink.push("fr", tone(1.9), 22050)   # nothing drained the ring: 3.8s into 2.0s
    assert sink.stats.dropped_phrases == 1
    assert any("ring full" in r.getMessage() for r in caplog.records)
    await sink.close()


# --- the real thing ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_real_output_device_opens_and_keeps_up():
    """
    Skipped where there is no PortAudio or no output device. Where there is
    one - this was written against Dante Virtual Soundcard - the stream
    opens by name, runs for half a second, and reports no underruns.
    """
    sd = pytest.importorskip("sounddevice")
    from lad_translate.session.localcard import list_output_devices

    try:
        devices = list_output_devices()
    except sd.PortAudioError:
        pytest.skip("PortAudio has no host API here")
    if not devices:
        pytest.skip("no output devices on this machine")
    card = devices[0]
    for d in devices:
        if "dante" in d.name.lower():
            card = d
    device = a_device((OutputChannel("fr", 1),), device_name=card.name,
                      sample_rate=int(card.default_samplerate), channel_count=card.max_output_channels)
    sink = LocalCardSink(device, LocalCardConfig(latency="high"))
    await sink.publish_languages(["fr"])
    await sink.push("fr", tone(0.3, amp=0.05), 22050)
    await asyncio.sleep(0.5)
    await sink.close()
    assert sink.card.name == card.name
    assert sink.stats.frames_out >= int(0.3 * card.default_samplerate)
    assert sink.stats.underruns == 0
