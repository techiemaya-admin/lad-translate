"""
push() must not overflow the publish queue.

An Arabic listener lost three phrases to this. The drift controller gates on
the queue depth BEFORE a phrase and never on what the phrase itself will add,
so with skip_at_s at 6s and a 12s queue there is 6s of headroom - and one
Arabic phrase can fill most of it. The next one overflowed, and LiveKit reports
an overflowing capture_frame as

    InvalidState - failed to capture frame

which reads like a transport fault rather than a full queue. The phrase was
lost either way; the misleading error just cost the time to find out why.

The rule this pins down: room.push may wait, and may drop and say so, but must
never hand the source more than it can hold.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

# livekit.rtc is not a test dependency - the model backends and the transport
# are both excluded from CI on purpose. push() needs exactly one symbol from
# it, AudioFrame, and it is a plain data carrier, so a stand-in is honest here
# rather than a mock of behaviour we then assert on.
if "livekit.rtc" not in sys.modules:
    class _AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

    _livekit = sys.modules.setdefault("livekit", types.ModuleType("livekit"))
    _rtc = types.ModuleType("livekit.rtc")
    _rtc.AudioFrame = _AudioFrame
    _livekit.rtc = _rtc
    sys.modules["livekit.rtc"] = _rtc

from lad_translate.session import room as room_mod


class FakeSource:
    """A publish queue that drains only when told to, and refuses overflow."""

    def __init__(self) -> None:
        self.queued_duration = 0.0
        self.captured_s: list[float] = []

    async def capture_frame(self, frame) -> None:
        seconds = frame.samples_per_channel / frame.sample_rate
        if self.queued_duration + seconds > room_mod.PLAYOUT_QUEUE_MS / 1000.0 + 1e-9:
            raise Exception("an RtcError occurred: InvalidState - failed to capture frame")
        self.queued_duration += seconds
        self.captured_s.append(seconds)

    def drain(self, seconds: float) -> None:
        self.queued_duration = max(0.0, self.queued_duration - seconds)


def _room_with(source):
    r = room_mod.TranslationRoom.__new__(room_mod.TranslationRoom)
    r.sample_rate = 16000
    r.num_channels = 1
    r._tracks = {"ar": room_mod.LanguageTrack(language="ar", source=source, track=None)}
    return r


def pcm_of(seconds: float, sample_rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


@pytest.mark.asyncio
async def test_push_waits_rather_than_overflowing():
    """The exact failure: a queue nearly full, then a phrase that will not fit."""
    source = FakeSource()
    r = _room_with(source)

    # Fill to just under capacity, the way two long phrases would.
    await r.push("ar", pcm_of(11.0), 16000)
    assert source.queued_duration == pytest.approx(11.0)

    # 2s more does not fit in a 12s queue. Drain while push is waiting.
    async def drain_soon():
        await asyncio.sleep(0.15)
        source.drain(6.0)

    asyncio.create_task(drain_soon())
    await r.push("ar", pcm_of(2.0), 16000)

    assert source.captured_s == pytest.approx([11.0, 2.0]), "the phrase was lost"
    assert source.queued_duration <= room_mod.PLAYOUT_QUEUE_MS / 1000.0


@pytest.mark.asyncio
async def test_push_drops_loudly_when_the_queue_never_drains(monkeypatch, caplog):
    """
    A track that has stopped draining must not wedge its language for ever.

    Dropping is the right outcome, but it has to be reported: this is exactly
    the case that previously surfaced as a transport error.
    """
    monkeypatch.setattr(room_mod, "PUSH_WAIT_TIMEOUT_S", 0.2)
    source = FakeSource()
    r = _room_with(source)
    await r.push("ar", pcm_of(11.5), 16000)

    with caplog.at_level("ERROR"):
        await r.push("ar", pcm_of(2.0), 16000)   # never fits, never drains

    assert source.captured_s == pytest.approx([11.5]), "should not have been captured"
    assert any("playout queue full" in rec.message for rec in caplog.records)
