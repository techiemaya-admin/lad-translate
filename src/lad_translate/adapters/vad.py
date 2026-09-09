"""
Streaming speech gate.

FastConformer has no silence handling of its own, and on a live phone that
produced text out of room tone: 24 and 48 second spans of invented words
between actual sentences, which filled the chunker until max_words and left a
listener waiting half a minute. The fixture never showed it, because clean read
prose has no room tone.

Silero, which faster-whisper already bundles, separates them clearly. Measured
on three seconds of each at 16kHz:

    speech      mean 0.783   80% of windows above 0.5
    silence     mean 0.002
    room tone   mean 0.010
    50Hz hum    mean 0.003

A fixed RMS threshold would have been quicker to write and wrong: the band that
matters is exactly where a quiet talker and a noisy room overlap, and an
amplitude test cannot tell them apart. That mistake already cost this project
once, in stt_whisper.py, where two thresholds that never met let the buffer grow
without bound.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from ..obs.log import get_logger

log = get_logger(__name__)

WINDOW = 512
"""Samples per Silero decision. 32ms at 16kHz, and the model's native size."""

CONTEXT = 64
"""Samples of history Silero keeps between windows. Its own default."""


class GateTimeline:
    """
    Maps a position in gated audio back to its position in the received stream.

    The gate removes silence, so anything timestamped against what came out of
    it is stamped in SPEECH seconds. Everything downstream counts RECEIVED
    seconds: obs.latency.AudioClock is anchored on room frames, which arrive
    whether or not anyone is talking. Hand one to the other and every removed
    second is reported as a second of latency - and since the error only ever
    accumulates, it grows for the length of the session.

    Measured live on 9 Sep 2026, before this existed. Glass-to-glass across
    five consecutive chunks:

        30.0s -> 36.7s -> 67.6s -> 89.7s -> 120.7s

    identical for fr, ar and de, while the machine sat at load 0.96 on 16
    vCPUs and shed no audio at all. Idle CPU plus monotone growth plus nothing
    dropped is an added constant that keeps growing, which is a clock rather
    than a slow stage. The re-anchor guard in session.pipeline cannot see it:
    that compares t_wall against t_audio on ROOM frames, and both of those
    advance normally. The gate is downstream of it and invisible to it.

    Held as breakpoints rather than a running total because the correction
    belongs to a POSITION and not to the moment of asking. A phrase that ends
    just before a pause must not be charged for that pause; a running total
    read at emit time would charge it. Only pauses create breakpoints, so an
    hour with forty of them costs forty entries.
    """

    __slots__ = ("_passed", "_suppressed")

    def __init__(self) -> None:
        self._passed: list[float] = []
        """Speech-seconds elapsed when each pause began. Non-decreasing."""

        self._suppressed: list[float] = []
        """Cumulative seconds removed by the end of that pause."""

    def hold(self, at_passed_s: float, seconds: float) -> None:
        """Record that `seconds` were removed after `at_passed_s` of speech."""
        if self._passed and self._passed[-1] == at_passed_s:
            self._suppressed[-1] += seconds
            return
        running = self._suppressed[-1] if self._suppressed else 0.0
        self._passed.append(at_passed_s)
        self._suppressed.append(running + seconds)

    def release(self, seconds: float) -> None:
        """
        Un-charge audio that was held back and then let through anyway.

        The pre-roll is suppressed window by window and only flushed once the
        gate opens, so without this every pause would over-report by up to the
        pre-roll length - 200ms each, which over a talk is the same kind of
        accumulating constant this class exists to remove.
        """
        if not self._suppressed:
            return
        floor = self._suppressed[-2] if len(self._suppressed) > 1 else 0.0
        self._suppressed[-1] = max(floor, self._suppressed[-1] - seconds)

    def received_time(self, speech_s: float) -> float:
        """Seconds of gated audio -> seconds of received audio."""
        # bisect_left, so a position sitting exactly on a breakpoint is NOT
        # charged for the pause that starts there: a phrase ending where the
        # speaker stopped arrived before the silence, not after it.
        i = bisect.bisect_left(self._passed, speech_s)
        return speech_s + (self._suppressed[i - 1] if i else 0.0)

    @property
    def suppressed_s(self) -> float:
        return self._suppressed[-1] if self._suppressed else 0.0

    def reset(self) -> None:
        self._passed.clear()
        self._suppressed.clear()


@dataclass
class GateStats:
    windows: int = 0
    speech_windows: int = 0
    passed_s: float = 0.0
    suppressed_s: float = 0.0

    @property
    def suppressed_fraction(self) -> float:
        total = self.passed_s + self.suppressed_s
        return self.suppressed_s / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "windows": self.windows,
            "speech_windows": self.speech_windows,
            "passed_s": round(self.passed_s, 2),
            "suppressed_s": round(self.suppressed_s, 2),
            "suppressed_fraction": round(self.suppressed_fraction, 3),
        }


class SpeechGate:
    """
    Passes speech through and holds silence back, one 32ms window at a time.

    Two details that decide whether it helps or hurts:

    PRE-ROLL. Gating on the first speech window clips the onset of the word that
    opened it, and a transducer given a truncated first phoneme produces a wrong
    first token that its own context then builds on. A short ring buffer of
    recent audio is flushed when speech starts, so the model sees the attack.

    HANGOVER. Closing the moment probability drops cuts the tail off every
    utterance and splits words across the pause inside them - "important"
    becomes two fragments. The gate stays open for a fixed time after the last
    speech window.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        pre_roll_ms: int = 200,
        hangover_ms: int = 400,
    ) -> None:
        if sample_rate != 16000:
            # Silero is trained at 16k and 8k. Anything else is a resample the
            # caller has skipped, and guessing here would hide their bug.
            raise ValueError(f"the gate needs 16kHz audio, not {sample_rate}")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.pre_roll = int(sample_rate * pre_roll_ms / 1000)
        self.hangover_windows = max(1, int((hangover_ms / 1000) * sample_rate / WINDOW))

        self._model = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._ring = np.zeros(0, dtype=np.float32)
        self._open = False
        self._quiet_windows = 0
        self.stats = GateStats()

        self.timeline = GateTimeline()
        """How to read the gate's output on the received clock. Callers that
        timestamp anything MUST map through this - see GateTimeline."""

    def _load(self):
        if self._model is None:
            from faster_whisper.vad import get_vad_model

            self._model = get_vad_model()
            log.info(
                "speech gate loaded",
                extra={
                    "threshold": self.threshold,
                    "pre_roll_ms": int(self.pre_roll / self.sample_rate * 1000),
                    "hangover_windows": self.hangover_windows,
                },
            )
        return self._model

    def feed(self, audio: np.ndarray) -> np.ndarray:
        """
        Returns the audio that should reach the model. Often empty.

        Whole windows only: a partial window is held until the next call rather
        than padded, because padding with zeros makes Silero see silence that
        was never in the stream.
        """
        model = self._load()
        self._pending = np.concatenate([self._pending, audio.astype(np.float32)])

        usable = (self._pending.size // WINDOW) * WINDOW
        if usable == 0:
            return np.zeros(0, dtype=np.float32)

        block, self._pending = self._pending[:usable], self._pending[usable:]
        scores = np.asarray(
            model(block, num_samples=WINDOW, context_size_samples=CONTEXT)
        ).ravel()

        out: list[np.ndarray] = []
        for i, score in enumerate(scores):
            window = block[i * WINDOW : (i + 1) * WINDOW]
            self.stats.windows += 1
            speech = bool(score >= self.threshold)
            if speech:
                self.stats.speech_windows += 1

            if speech:
                if not self._open:
                    # Opening: flush the pre-roll so the word's attack survives.
                    if self._ring.size:
                        held = self._ring.size / self.sample_rate
                        out.append(self._ring)
                        self.stats.passed_s += held
                        # It was counted as suppressed on the way in and is
                        # being let through after all, so give it back to both
                        # the clock and the stats rather than counting it twice.
                        self.stats.suppressed_s -= held
                        self.timeline.release(held)
                        self._ring = np.zeros(0, dtype=np.float32)
                    self._open = True
                self._quiet_windows = 0
                out.append(window)
                self.stats.passed_s += WINDOW / self.sample_rate
            elif self._open:
                self._quiet_windows += 1
                if self._quiet_windows <= self.hangover_windows:
                    # Still inside an utterance; a pause is not an ending.
                    out.append(window)
                    self.stats.passed_s += WINDOW / self.sample_rate
                else:
                    self._open = False
                    self._suppress(window)
            else:
                self._suppress(window)

        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def _suppress(self, window: np.ndarray) -> None:
        """
        Hold one window back, and tell the timeline it happened.

        Every path that drops audio goes through here. Recording the drop next
        to the drop is the point: the original bug was that the gate removed
        time and nothing downstream was told, and a second removal path that
        forgot to update the clock would recreate it exactly.
        """
        self._remember(window)
        seconds = WINDOW / self.sample_rate
        self.stats.suppressed_s += seconds
        self.timeline.hold(self.stats.passed_s, seconds)

    def _remember(self, window: np.ndarray) -> None:
        """Keep the most recent pre_roll samples, and no more."""
        self._ring = np.concatenate([self._ring, window])[-self.pre_roll :]

    def reset(self) -> None:
        """
        Forget the stream so far.

        transcribe() builds a fresh geometry and schedule for every stream, so
        its audio positions restart at zero. A gate carried over from a
        previous stream would map those against the old stream's pauses.
        """
        self._pending = np.zeros(0, dtype=np.float32)
        self._ring = np.zeros(0, dtype=np.float32)
        self._open = False
        self._quiet_windows = 0
        self.stats = GateStats()
        self.timeline.reset()
