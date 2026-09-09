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

from dataclasses import dataclass

import numpy as np

from ..obs.log import get_logger

log = get_logger(__name__)

WINDOW = 512
"""Samples per Silero decision. 32ms at 16kHz, and the model's native size."""

CONTEXT = 64
"""Samples of history Silero keeps between windows. Its own default."""


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
                        out.append(self._ring)
                        self.stats.passed_s += self._ring.size / self.sample_rate
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
                    self._remember(window)
                    self.stats.suppressed_s += WINDOW / self.sample_rate
            else:
                self._remember(window)
                self.stats.suppressed_s += WINDOW / self.sample_rate

        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def _remember(self, window: np.ndarray) -> None:
        """Keep the most recent pre_roll samples, and no more."""
        self._ring = np.concatenate([self._ring, window])[-self.pre_roll :]
