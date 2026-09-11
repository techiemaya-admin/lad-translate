"""
Where translated audio goes.

Until now there was one destination and the pipeline named it directly:
`session.room.push(...)`. A venue with an infrared system needs the same audio
on a second path at the same time, so the destination becomes an interface and
TranslationRoom becomes one implementation of it.

session/room.py already has the shape -- publish_languages, push, queue_depth,
close -- so it satisfies AudioSink without changes. That is deliberate: the
interface was read off the working code rather than imposed on it.

Two things are worth stating before a hardware sink is written against this.

FAILURE IS NOT SYMMETRIC. The phones are the product; the IR rig is an
addition. A Dante device that disappears mid-session must not take the WebRTC
audience down with it, so FanOutSink raises only for its primary sink and
degrades for the rest. The reverse -- letting a secondary failure end the
session -- would mean adding hardware output makes the service less reliable
than not having it.

QUEUE DEPTH IS A MAXIMUM, NOT A SUM. session/drift.py steers on how far behind
playout is. With two sinks there are two queues, and the audience that matters
is whichever is further behind: handsets drifting while WebRTC is healthy is
still an audience out of sync. Summing would double-count the same phrase and
make the controller skip far too eagerly.

A clocked device also behaves differently from LiveKit's AudioSource. The SFU
is pushed when there is speech and sends nothing between phrases; a Dante card
consumes a sample every 1/48000s forever and must be fed silence when nobody is
talking. A sink for one is therefore a ring buffer plus a pump, not a thin
wrapper, and its queue_depth is the fill of that buffer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..obs.log import get_logger

log = get_logger(__name__)


@runtime_checkable
class AudioSink(Protocol):
    """One destination for synthesised speech."""

    async def publish_languages(self, languages: list[str]) -> None:
        """Open a stream per language. Called once, before any push."""
        ...

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        """Hand one phrase of 16-bit PCM to a language's stream."""
        ...

    def queue_depth(self, language: str) -> float:
        """Seconds synthesised but not yet heard. The drift signal."""
        ...

    async def close(self) -> None:
        ...


class FanOutSink:
    """
    One primary sink and any number of secondary ones.

    The primary's errors propagate; a secondary's are logged and counted. A
    secondary that fails is not retried or reopened here -- reconnecting a
    sound card is the sink's own business, and a fan-out that quietly swallowed
    a dead device for an hour would report success for an audience hearing
    nothing.
    """

    def __init__(self, primary: AudioSink, *secondary: AudioSink) -> None:
        self.primary = primary
        self.secondary = list(secondary)
        self.failures: dict[int, int] = {}
        """Failed pushes per secondary sink, by position. Reported at close."""

    @property
    def sinks(self) -> list[AudioSink]:
        return [self.primary, *self.secondary]

    async def publish_languages(self, languages: list[str]) -> None:
        await self.primary.publish_languages(languages)
        for index, sink in enumerate(self.secondary):
            try:
                await sink.publish_languages(languages)
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                # A secondary that cannot open is dropped for the session
                # rather than retried per phrase. Opening a device fails for
                # structural reasons -- wrong name, wrong channel count, card
                # not present -- and none of them fix themselves mid-event.
                self._note_failure(index, sink, exc, "publish")

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        await self.primary.push(language, pcm, sample_rate)
        for index, sink in enumerate(self.secondary):
            try:
                await sink.push(language, pcm, sample_rate)
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                self._note_failure(index, sink, exc, "push", language)

    def queue_depth(self, language: str) -> float:
        """The furthest-behind queue across all sinks. See the module docstring."""
        depths = [self.primary.queue_depth(language)]
        for index, sink in enumerate(self.secondary):
            try:
                depths.append(sink.queue_depth(language))
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                # Drift control runs on this. A sink that cannot report its
                # depth is excluded rather than counted as zero, which would
                # read as a healthy queue and suppress the correction.
                self._note_failure(index, sink, exc, "queue_depth", language)
        return max(depths)

    async def close(self) -> None:
        for index, sink in enumerate(self.secondary):
            try:
                await sink.close()
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                self._note_failure(index, sink, exc, "close")
        if self.failures:
            log.warning(
                "secondary sinks reported failures this session",
                extra={"failures": {str(k): v for k, v in self.failures.items()}},
            )
        # Last, and outside the guard: the primary closing badly is a real
        # error and the session should hear about it.
        await self.primary.close()

    def _note_failure(
        self, index: int, sink: AudioSink, exc: Exception, op: str, language: str = ""
    ) -> None:
        count = self.failures.get(index, 0) + 1
        self.failures[index] = count
        # Logged at every failure, not sampled: a venue post mortem needs to
        # see when a device started failing, not just that it did.
        log.error(
            "secondary audio sink failed",
            extra={
                "sink": type(sink).__name__,
                "sink_index": index,
                "operation": op,
                "language": language,
                "failures": count,
                "error": str(exc),
            },
        )
