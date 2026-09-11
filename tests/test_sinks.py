"""
Fan-out to more than one audio destination.

The property that matters: adding hardware output must not make the WebRTC
audience less reliable than not having it. A Dante card that disappears
mid-session is a degraded event, not a failed one.
"""

from __future__ import annotations

import pytest

from lad_translate.session.sinks import AudioSink, FanOutSink


class FakeSink:
    """Records what it was asked to do, and fails on demand."""

    def __init__(self, depth: float = 0.0, fail_on: set[str] | None = None) -> None:
        self.depth = depth
        self.fail_on = fail_on or set()
        self.published: list[str] = []
        self.pushed: list[tuple[str, bytes]] = []
        self.closed = False

    def _maybe_fail(self, op: str) -> None:
        if op in self.fail_on:
            raise RuntimeError(f"{op} failed")

    async def publish_languages(self, languages: list[str]) -> None:
        self._maybe_fail("publish")
        self.published = list(languages)

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        self._maybe_fail("push")
        self.pushed.append((language, pcm))

    def queue_depth(self, language: str) -> float:
        self._maybe_fail("queue_depth")
        return self.depth

    async def close(self) -> None:
        self._maybe_fail("close")
        self.closed = True


def test_fake_sink_satisfies_the_protocol():
    """If the fake drifts from the interface these tests stop meaning anything."""
    assert isinstance(FakeSink(), AudioSink)


async def test_audio_reaches_every_sink():
    primary, secondary = FakeSink(), FakeSink()
    fan = FanOutSink(primary, secondary)

    await fan.publish_languages(["fr", "ar"])
    await fan.push("fr", b"\x00\x01", 22050)

    assert primary.published == secondary.published == ["fr", "ar"]
    assert primary.pushed == secondary.pushed == [("fr", b"\x00\x01")]


async def test_a_failing_secondary_does_not_break_the_primary():
    """The phones are the product; the IR rig is an addition."""
    primary, secondary = FakeSink(), FakeSink(fail_on={"push"})
    fan = FanOutSink(primary, secondary)

    await fan.push("fr", b"\x00\x01", 22050)

    assert primary.pushed == [("fr", b"\x00\x01")]
    assert fan.failures == {0: 1}


async def test_a_failing_primary_is_raised():
    """A dead WebRTC path is a real failure and the session must hear about it."""
    fan = FanOutSink(FakeSink(fail_on={"push"}), FakeSink())
    with pytest.raises(RuntimeError):
        await fan.push("fr", b"\x00\x01", 22050)


async def test_a_secondary_that_cannot_open_is_counted_at_publish():
    primary, secondary = FakeSink(), FakeSink(fail_on={"publish"})
    fan = FanOutSink(primary, secondary)

    await fan.publish_languages(["fr"])

    assert primary.published == ["fr"]
    assert secondary.published == []
    assert fan.failures == {0: 1}


def test_queue_depth_is_the_furthest_behind_sink():
    """
    Handsets drifting while WebRTC is healthy is still an audience out of sync,
    so drift control steers on the worst queue.
    """
    fan = FanOutSink(FakeSink(depth=0.5), FakeSink(depth=2.4))
    assert fan.queue_depth("fr") == 2.4


def test_queue_depth_is_not_a_sum():
    """Summing double-counts one phrase and makes the controller skip too eagerly."""
    fan = FanOutSink(FakeSink(depth=1.0), FakeSink(depth=1.0))
    assert fan.queue_depth("fr") == 1.0


def test_a_sink_that_cannot_report_depth_is_excluded_not_read_as_zero():
    """Zero would read as a healthy queue and suppress the correction."""
    fan = FanOutSink(FakeSink(depth=3.0), FakeSink(fail_on={"queue_depth"}))
    assert fan.queue_depth("fr") == 3.0
    assert fan.failures == {0: 1}


async def test_close_reaches_every_sink_even_when_one_fails():
    primary, bad, good = FakeSink(), FakeSink(fail_on={"close"}), FakeSink()
    fan = FanOutSink(primary, bad, good)

    await fan.close()

    assert primary.closed and good.closed
    assert not bad.closed
    assert fan.failures == {0: 1}


async def test_a_lone_primary_behaves_exactly_as_before():
    """A session with no hardware output must be unchanged by any of this."""
    primary = FakeSink(depth=1.25)
    fan = FanOutSink(primary)

    await fan.publish_languages(["fr"])
    await fan.push("fr", b"\x00", 22050)
    await fan.close()

    assert fan.queue_depth("fr") == 1.25
    assert primary.closed
    assert fan.failures == {}
