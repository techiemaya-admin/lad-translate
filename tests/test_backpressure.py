"""
Backlog guard tests.

Written against the failure actually measured: feeding a 25.6s clip at real
speed with a backend that could not keep up gave p50 90s and p95 168s, growing
monotonically. The guard's job is to make that impossible.
"""

from __future__ import annotations

import asyncio

import pytest

from lad_translate.adapters.base import AudioFrame
from lad_translate.session.backpressure import BacklogGuard, guarded

RATE = 16_000
FRAME_S = 0.1
FRAME_BYTES = b"\x00\x00" * int(RATE * FRAME_S)


def frame(index: int) -> AudioFrame:
    return AudioFrame(
        pcm=FRAME_BYTES, sample_rate=RATE, t_audio=index * FRAME_S, t_wall=index * FRAME_S
    )


def test_rejects_a_recovery_target_that_cannot_settle():
    """Shedding only to the threshold makes the next frame breach it again."""
    with pytest.raises(ValueError, match="never settles"):
        BacklogGuard(max_lag_s=1.0, recover_to_s=1.0)


def test_no_shedding_while_the_consumer_keeps_up():
    guard = BacklogGuard(max_lag_s=3.0, recover_to_s=1.0)
    for i in range(50):
        guard.push(frame(i))
        list(_drain_available(guard))
    assert guard.stats.frames_dropped == 0
    assert guard.lag_seconds == 0.0


def test_backlog_is_capped_rather_than_growing_without_bound():
    """The measured failure: nothing consumes, so the queue must not run away."""
    guard = BacklogGuard(max_lag_s=3.0, recover_to_s=1.0)
    for i in range(200):  # 20 seconds of audio, nothing draining
        guard.push(frame(i))
    assert guard.lag_seconds <= 3.0, "backlog grew past the threshold"
    assert guard.stats.frames_dropped > 0
    assert guard.stats.seconds_dropped > 0


def test_shedding_recovers_well_below_the_threshold():
    """
    Checked immediately after a shed, not at an arbitrary point.

    Between sheds the lag is meant to climb back up towards max_lag_s. That
    hysteresis is the point: it is what stops the guard dropping a frame on
    every push. The always-true invariant is lag <= max_lag_s, covered by
    test_backlog_is_capped_rather_than_growing_without_bound.
    """
    guard = BacklogGuard(max_lag_s=3.0, recover_to_s=1.0)
    for i in range(100):
        before = guard.stats.shed_events
        guard.push(frame(i))
        if guard.stats.shed_events > before:
            assert guard.lag_seconds <= 1.0 + FRAME_S
            return
    pytest.fail("no shed event occurred")


def test_lag_climbs_between_sheds_rather_than_dropping_every_frame():
    guard = BacklogGuard(max_lag_s=3.0, recover_to_s=1.0)
    lags = []
    for i in range(100):
        guard.push(frame(i))
        lags.append(guard.lag_seconds)
    assert max(lags) <= 3.0
    assert max(lags) > 1.0 + FRAME_S, "lag never recovered upward; guard is thrashing"


def test_shedding_is_occasional_not_per_frame():
    """One clean cut sounds far better than a drop on every frame."""
    guard = BacklogGuard(max_lag_s=3.0, recover_to_s=1.0)
    for i in range(200):
        guard.push(frame(i))
    assert guard.stats.shed_events < 20, (
        f"{guard.stats.shed_events} shed events for 200 frames is thrashing"
    )


def test_dropped_audio_is_counted_not_silently_discarded():
    guard = BacklogGuard(max_lag_s=2.0, recover_to_s=0.5)
    for i in range(150):
        guard.push(frame(i))
    summary = guard.summary()
    assert summary["frames_dropped"] > 0
    assert summary["seconds_dropped"] > 0
    assert summary["drop_rate"] > 0
    assert summary["peak_lag_s"] >= 2.0


def test_frames_survive_in_order_when_nothing_is_shed():
    guard = BacklogGuard()
    for i in range(5):
        guard.push(frame(i))
    got = [f.t_audio for f in _drain_available(guard)]
    assert got == [i * FRAME_S for i in range(5)]


def test_oldest_audio_goes_first_so_the_stream_catches_up():
    guard = BacklogGuard(max_lag_s=1.0, recover_to_s=0.3)
    for i in range(50):
        guard.push(frame(i))
    remaining = [f.t_audio for f in _drain_available(guard)]
    assert remaining, "the buffer must never be emptied completely"
    assert remaining[-1] == 49 * FRAME_S, "the newest frame must be kept"


async def test_guarded_wrapper_passes_everything_through_when_keeping_up():
    async def source():
        for i in range(20):
            yield frame(i)
            await asyncio.sleep(0)

    guard = BacklogGuard()
    got = [f.t_audio async for f in guarded(source(), guard)]
    assert len(got) == 20
    assert guard.stats.frames_dropped == 0


async def test_guarded_wrapper_terminates_when_the_source_ends():
    async def source():
        for i in range(3):
            yield frame(i)

    guard = BacklogGuard()
    got = [f async for f in guarded(source(), guard)]
    assert len(got) == 3
    assert guard.stats.frames_out == 3


def _drain_available(guard: BacklogGuard):
    """Synchronously pull whatever is buffered right now."""
    while guard._buffer:
        guard.stats.frames_out += 1
        yield guard._buffer.popleft()
