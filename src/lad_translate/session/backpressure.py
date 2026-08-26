"""
Audio backlog control.

Measured on the dev Mac: feeding a 25.6 second clip at real speed with a
backend that could not keep up produced a p50 latency of 90 seconds and a p95
of 168 seconds. The lag grew monotonically and never recovered, because nothing
in the pipeline noticed it was behind.

That is a design gap, not a hardware problem. A faster backend widens the
margin; it does not add a floor. Any stall long enough to build a queue starts
the same runaway, and a listener three minutes behind the speaker has no
product at all.

The policy here is deliberate and it loses audio on purpose:

    Being current matters more than being complete.

When the backlog passes the threshold, the oldest audio is dropped so the
stream catches up to the speaker. The audience misses a few seconds rather than
sliding permanently behind. Every drop is counted and logged, because silently
discarding a speaker's words is exactly the kind of thing that must show up on
a dashboard rather than in a complaint after the event.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..adapters.base import AudioFrame
from ..obs.log import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class BacklogStats:
    frames_in: int = 0
    frames_out: int = 0
    frames_dropped: int = 0
    seconds_dropped: float = 0.0
    shed_events: int = 0
    peak_lag_s: float = 0.0

    @property
    def drop_rate(self) -> float:
        return self.frames_dropped / self.frames_in if self.frames_in else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "frames_dropped": self.frames_dropped,
            "seconds_dropped": round(self.seconds_dropped, 2),
            "shed_events": self.shed_events,
            "peak_lag_s": round(self.peak_lag_s, 2),
            "drop_rate": round(self.drop_rate, 4),
        }


class BacklogGuard:
    """
    Bounded buffer between the room and the STT backend.

    Sits in the audio path as a passthrough that can shed. Producer and
    consumer run at their own speeds; when the gap opens past `max_lag_s` the
    oldest frames go.
    """

    def __init__(
        self,
        max_lag_s: float = 3.0,
        recover_to_s: float = 1.0,
        warn_lag_s: float = 1.5,
    ) -> None:
        if recover_to_s >= max_lag_s:
            raise ValueError("recover_to_s must be below max_lag_s or shedding never settles")
        self.max_lag_s = max_lag_s
        """Backlog at which audio starts being discarded."""

        self.recover_to_s = recover_to_s
        """
        Shed down to this, not just to below max_lag_s.

        Trimming to the threshold means the next frame crosses it again and the
        guard sheds on almost every frame, which sounds far worse than one
        clean cut.
        """

        self.warn_lag_s = warn_lag_s
        self.stats = BacklogStats()
        self._buffer: deque[AudioFrame] = deque()
        self._arrived = asyncio.Event()
        self._closed = False
        self._newest_audio = 0.0
        self._warned = False

    # -------------------------------------------------------------------------

    @property
    def lag_seconds(self) -> float:
        """Audio time between the newest frame received and the oldest buffered."""
        if not self._buffer:
            return 0.0
        return max(0.0, self._newest_audio - self._buffer[0].t_audio)

    def push(self, frame: AudioFrame) -> None:
        """Accept one frame, shedding older audio if the backlog has opened up."""
        self.stats.frames_in += 1
        self._newest_audio = frame.t_audio + frame.duration
        self._buffer.append(frame)
        self._shed_if_behind()
        self._arrived.set()

    def close(self) -> None:
        """Signal that no more frames are coming."""
        self._closed = True
        self._arrived.set()

    async def feed(self, frames: AsyncIterator[AudioFrame]) -> None:
        """Consume the source stream into the buffer. Run as a task."""
        try:
            async for frame in frames:
                self.push(frame)
        finally:
            self.close()

    async def drain(self) -> AsyncIterator[AudioFrame]:
        """Yield frames to the backend, oldest first, minus anything shed."""
        while True:
            while self._buffer:
                self.stats.frames_out += 1
                yield self._buffer.popleft()
            if self._closed:
                return
            self._arrived.clear()
            await self._arrived.wait()

    # -------------------------------------------------------------------------

    def _shed_if_behind(self) -> None:
        lag = self.lag_seconds
        self.stats.peak_lag_s = max(self.stats.peak_lag_s, lag)

        if lag >= self.warn_lag_s and not self._warned:
            self._warned = True
            log.warning(
                "audio backlog building",
                extra={"lag_s": round(lag, 2), "threshold_s": self.max_lag_s},
            )
        elif lag < self.warn_lag_s:
            self._warned = False

        if lag < self.max_lag_s:
            return

        dropped = 0
        seconds = 0.0
        # Always leave the newest frame: shedding the whole buffer would stall
        # the backend instead of catching it up.
        while len(self._buffer) > 1 and self.lag_seconds > self.recover_to_s:
            frame = self._buffer.popleft()
            dropped += 1
            seconds += frame.duration

        if not dropped:
            return
        self.stats.frames_dropped += dropped
        self.stats.seconds_dropped += seconds
        self.stats.shed_events += 1
        log.error(
            "audio dropped to recover from backlog",
            extra={
                "frames_dropped": dropped,
                "seconds_dropped": round(seconds, 2),
                "lag_before_s": round(lag, 2),
                "lag_after_s": round(self.lag_seconds, 2),
                "total_seconds_dropped": round(self.stats.seconds_dropped, 2),
            },
        )

    def summary(self) -> dict[str, float | int]:
        summary = self.stats.as_dict()
        if self.stats.frames_dropped:
            log.warning("session dropped audio", extra=summary)
        return summary


async def guarded(
    frames: AsyncIterator[AudioFrame], guard: BacklogGuard
) -> AsyncIterator[AudioFrame]:
    """
    Wrap a frame stream with a guard, as a single async iterator.

    Convenience for the common case where the caller does not want to manage
    the feeder task itself.
    """
    feeder = asyncio.create_task(guard.feed(frames))
    try:
        async for frame in guard.drain():
            yield frame
        await feeder
    finally:
        if not feeder.done():
            feeder.cancel()
