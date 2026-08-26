"""
Glass to glass latency measurement.

VOAG has none of this. It disables OpenTelemetry exports outright
(agent/worker.py:40) and its only latency code is two offline benchmark
scripts. That is why nobody there can say what the p95 is. This project
measures from the first session onward.

The number that matters:

    glass_to_glass = wall time the translated audio was published
                   - wall time the source audio for the end of that phrase arrived

Measured per chunk per language. The stage breakdown alongside it says WHERE
the budget went, which is what makes it actionable rather than just alarming.

Everything is recorded against time.monotonic(). Wall-clock time is not used
anywhere in the measurement path because it can step backwards.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .log import get_logger

log = get_logger(__name__)


class Stage(str, Enum):
    """Points a chunk passes through. Recorded in this order."""

    COMMITTED = "committed"
    """Chunker released the phrase."""

    TRANSLATED = "translated"
    """Translation returned, per language."""

    TTS_FIRST_AUDIO = "tts_first_audio"
    """First synthesised audio chunk available. Not the last: the audience
    hears the first one, and total synthesis time is irrelevant to them."""

    PUBLISHED = "published"
    """First audio frame handed to the room, per language."""


class AudioClock:
    """
    Maps a position in the source audio to the wall time it arrived.

    A live stream arrives in real time, so the mapping is a fixed offset. It is
    established from the first frame and never adjusted, which means clock drift
    on the publisher shows up in the latency figures rather than being hidden.
    """

    __slots__ = ("_epoch",)

    def __init__(self) -> None:
        self._epoch: float | None = None

    def anchor(self, t_audio: float, t_wall: float) -> None:
        """Set the mapping from the first audio frame of the session."""
        if self._epoch is None:
            self._epoch = t_wall - t_audio

    @property
    def anchored(self) -> bool:
        return self._epoch is not None

    def wall_for(self, t_audio: float) -> float:
        if self._epoch is None:
            raise RuntimeError("AudioClock used before the first frame arrived")
        return self._epoch + t_audio


@dataclass(slots=True)
class ChunkTrace:
    """Timings for one chunk in one target language."""

    chunk_id: int
    language: str
    audio_end_wall: float
    """When the speaker finished this phrase, in monotonic time."""

    marks: dict[Stage, float] = field(default_factory=dict)

    def mark(self, stage: Stage, t_wall: float) -> None:
        # First mark wins. A stage that fires twice is a retry, and the
        # audience heard the first one.
        self.marks.setdefault(stage, t_wall)

    @property
    def glass_to_glass(self) -> float | None:
        published = self.marks.get(Stage.PUBLISHED)
        return None if published is None else published - self.audio_end_wall

    def breakdown(self) -> dict[str, float]:
        """Per-stage durations in seconds. Missing stages are omitted."""
        out: dict[str, float] = {}
        committed = self.marks.get(Stage.COMMITTED)
        if committed is not None:
            out["chunker"] = committed - self.audio_end_wall
        pairs = (
            (Stage.COMMITTED, Stage.TRANSLATED, "translate"),
            (Stage.TRANSLATED, Stage.TTS_FIRST_AUDIO, "tts"),
            (Stage.TTS_FIRST_AUDIO, Stage.PUBLISHED, "publish"),
        )
        for start, end, label in pairs:
            a, b = self.marks.get(start), self.marks.get(end)
            if a is not None and b is not None:
                out[label] = b - a
        return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass(slots=True)
class LanguageStats:
    language: str
    count: int
    p50: float
    p95: float
    p99: float
    worst: float
    stage_means: dict[str, float]

    def breached(self, slo_seconds: float) -> bool:
        return self.p95 > slo_seconds


class LatencyRecorder:
    """
    Collects traces for a session and reports percentiles.

    Sample cap exists so a three hour event cannot grow this without bound.
    Beyond the cap the oldest samples are dropped, which biases the figures
    towards the recent part of the session. That is the right bias for a live
    event: what matters is whether it is drifting now.
    """

    def __init__(self, slo_seconds: float = 2.0, sample_cap: int = 20_000) -> None:
        self.slo_seconds = slo_seconds
        self.clock = AudioClock()
        self._sample_cap = sample_cap
        self._open: dict[tuple[int, str], ChunkTrace] = {}
        self._g2g: dict[str, list[float]] = defaultdict(list)
        self._stages: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._breaches = 0
        self._revisions = 0

    # -------------------------------------------------------------------------

    def open_chunk(self, chunk_id: int, language: str, t_audio_end: float) -> ChunkTrace:
        trace = ChunkTrace(
            chunk_id=chunk_id,
            language=language,
            audio_end_wall=self.clock.wall_for(t_audio_end),
        )
        self._open[(chunk_id, language)] = trace
        return trace

    def mark(self, chunk_id: int, language: str, stage: Stage, t_wall: float) -> None:
        trace = self._open.get((chunk_id, language))
        if trace is None:
            log.warning(
                "latency mark for unknown chunk",
                extra={"chunk_id": chunk_id, "language": language, "stage": stage.value},
            )
            return
        trace.mark(stage, t_wall)
        if stage is Stage.PUBLISHED:
            self._close(trace)

    def record_revision(self, chunk_id: int) -> None:
        """A committed chunk was later contradicted by the STT backend."""
        self._revisions += 1
        log.warning("committed text was revised", extra={"chunk_id": chunk_id})

    # -------------------------------------------------------------------------

    def _close(self, trace: ChunkTrace) -> None:
        g2g = trace.glass_to_glass
        if g2g is None:
            return
        self._append(self._g2g[trace.language], g2g)
        for label, seconds in trace.breakdown().items():
            self._append(self._stages[trace.language][label], seconds)

        breached = g2g > self.slo_seconds
        if breached:
            self._breaches += 1
        log.info(
            "chunk published",
            extra={
                "chunk_id": trace.chunk_id,
                "language": trace.language,
                "glass_to_glass_s": round(g2g, 3),
                "slo_breach": breached,
                **{f"stage_{k}_s": round(v, 3) for k, v in trace.breakdown().items()},
            },
        )
        self._open.pop((trace.chunk_id, trace.language), None)

    def _append(self, target: list[float], value: float) -> None:
        target.append(value)
        if len(target) > self._sample_cap:
            del target[: len(target) - self._sample_cap]

    # -------------------------------------------------------------------------

    def stats(self, language: str) -> LanguageStats:
        values = self._g2g[language]
        return LanguageStats(
            language=language,
            count=len(values),
            p50=percentile(values, 0.50),
            p95=percentile(values, 0.95),
            p99=percentile(values, 0.99),
            worst=max(values) if values else 0.0,
            stage_means={
                label: statistics.fmean(samples)
                for label, samples in self._stages[language].items()
                if samples
            },
        )

    def summary(self) -> dict[str, object]:
        """Session-level report. Log this at end of session, and periodically."""
        per_language = {lang: self.stats(lang) for lang in self._g2g}
        return {
            "slo_seconds": self.slo_seconds,
            "slo_breaches": self._breaches,
            "revisions_after_commit": self._revisions,
            "unfinished_chunks": len(self._open),
            "languages": {
                lang: {
                    "count": s.count,
                    "p50_s": round(s.p50, 3),
                    "p95_s": round(s.p95, 3),
                    "p99_s": round(s.p99, 3),
                    "worst_s": round(s.worst, 3),
                    "breached": s.breached(self.slo_seconds),
                    "stages": {k: round(v, 3) for k, v in s.stage_means.items()},
                }
                for lang, s in per_language.items()
            },
        }
