"""
The source stream must end when the speaker does.

livekit-rtc ends an AudioStream only on an end-of-stream event from the native
side, and when the speaker left that event did not come. The iterator blocked
for ever, so the pipeline never reached chunker.flush(): the medical script
lost its closing 15 words, and the session sat live for minutes after the
speaker had gone.

These tests stand in a stream that does exactly that - yields some audio, then
blocks for ever - and pin down that the room ends it on the track events, and
that a speaker who rejoins in time carries on the same talk.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lad_translate.session.room import TranslationRoom

FRAME_S = 0.02


def _event(i: int):
    samples = int(48_000 * FRAME_S)
    return SimpleNamespace(
        frame=SimpleNamespace(data=bytes([i]) * 2 * samples, sample_rate=48_000,
                              samples_per_channel=samples)
    )


class _StuckStream:
    """Yields its frames, then blocks for ever, as the real stream did."""

    def __init__(self, frames: int, hang: bool = True) -> None:
        self._left = frames
        self._hang = hang
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._left:
            self._left -= 1
            await asyncio.sleep(0)
            return _event(self._left)
        if not self._hang:
            raise StopAsyncIteration
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def _room(streams: dict) -> TranslationRoom:
    room = TranslationRoom("test-room", sample_rate=22_050, source_timeout_s=1.0)
    room.source_rejoin_s = 0.2
    room._open_stream = lambda track: streams[track]
    return room


def _subscribe(room: TranslationRoom, track: str) -> None:
    """What the track_subscribed handler does."""
    room._source_track = track
    room._source_gone.clear()
    room._source_ready.set()


async def _collect(room: TranslationRoom, on_frame=None) -> list:
    frames = []
    async for frame in room.source_frames():
        frames.append(frame)
        if on_frame:
            on_frame(len(frames))
    return frames


async def test_stream_ends_when_the_speaker_leaves():
    stream = _StuckStream(3)
    room = _room({"t1": stream})
    _subscribe(room, "t1")

    def leave_after_last(n):
        if n == 3:
            asyncio.get_running_loop().call_soon(room._end_source, "speaker")

    frames = await asyncio.wait_for(_collect(room, leave_after_last), timeout=2.0)

    assert len(frames) == 3
    assert [round(f.t_audio, 3) for f in frames] == [0.0, 0.02, 0.04]
    await asyncio.sleep(0)
    assert stream.closed


async def test_a_stream_that_ends_by_itself_is_not_reopened():
    opened = []
    room = _room({})
    room._open_stream = lambda track: opened.append(track) or _StuckStream(2, hang=False)
    _subscribe(room, "t1")

    frames = await asyncio.wait_for(_collect(room), timeout=2.0)

    assert len(frames) == 2
    assert opened == ["t1"]


async def test_a_speaker_who_rejoins_in_time_continues_the_same_talk():
    room = _room({"t1": _StuckStream(2), "t2": _StuckStream(2)})
    room.source_rejoin_s = 1.0
    _subscribe(room, "t1")
    loop = asyncio.get_running_loop()

    def drop_and_rejoin(n):
        if n == 2:
            loop.call_soon(room._end_source, "speaker")
            loop.call_later(0.05, _subscribe, room, "t2")
        if n == 4:
            loop.call_soon(room._end_source, "speaker")

    frames = await asyncio.wait_for(_collect(room, drop_and_rejoin), timeout=3.0)

    assert len(frames) == 4
    # The clock carries on across the rejoin rather than restarting at zero.
    assert [round(f.t_audio, 3) for f in frames] == [0.0, 0.02, 0.04, 0.06]


async def test_ending_twice_is_harmless():
    room = _room({})
    _subscribe(room, "t1")
    room._end_source("speaker")
    room._end_source("speaker")
    assert room._source_track is None
    assert not room._source_ready.is_set()
