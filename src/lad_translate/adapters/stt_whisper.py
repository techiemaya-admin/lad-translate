"""
faster-whisper speech to text. DEVELOPMENT ONLY.

Whisper is not a streaming model. It is a 30 second window encoder-decoder, so
every "streaming Whisper" is really a sliding buffer re-transcribed on a timer.
This adapter is that, and it is honest about it.

Measured against the chunker in tools/chunker_replay.py, a Whisper-class
backend spends about 3.2 seconds at p50 before the chunker can commit, against
a total product budget of 2 seconds. That gap is architectural, not
computational: the sliding window leaves a long unstable tail and
LocalAgreement has to wait through it. A faster GPU does not close it.

So this exists to exercise the pipeline on a machine with no CUDA. It reports
latency_credible=False and any session using it must never be quoted as
product latency. Production wants a streaming transducer: NVIDIA's cache-aware
streaming FastConformer, or Deepgram on-prem.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Self

import numpy as np

from ..obs.log import get_logger
from .base import AudioFrame, Hypothesis, SttAdapter

log = get_logger(__name__)

WHISPER_SAMPLE_RATE = 16_000

# Phrases Whisper emits when there is nothing to transcribe. They come from its
# training data, which is full of YouTube audio, and they appear over silence
# and room tone with high confidence in the text itself.
#
# Observed live: "Thanks for watching!" and a long run of bare "Thank you."
# in a session where nobody said either. Translated and spoken to an audience,
# an invented politeness is worse than a gap.
#
# Matched only when the segment ALSO looks like non-speech. "Thank you" is a
# real thing people say at a conference, and a blanket blocklist would delete
# genuine speech to remove an artefact.
HALLUCINATED_ON_SILENCE = frozenset(
    {
        "thank you.", "thank you", "thanks for watching!", "thanks for watching",
        "thank you for watching.", "thank you for watching",
        "thanks for watching and see you next time.",
        "please subscribe to my channel.", "subtitles by the amara.org community",
        "you", "bye.", "bye", "okay.", ".", "...",
    }
)


def is_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    """
    Whether a segment looks invented rather than heard.

    Three independent signals, because none is sufficient alone:

      no_speech_prob   the model's own verdict that this was not speech. High
                       and yet it produced words anyway is the exact shape of
                       a silence hallucination.
      avg_logprob      confidence. Real speech transcribed badly still scores
                       higher than text conjured from nothing.
      the phrase       only consulted when one of the above is already
                       suspicious, so genuine thanks survive.
    """
    stripped = text.strip().lower()
    if not stripped:
        return True

    # Confidently non-speech: drop whatever it produced, whatever the words.
    if no_speech_prob > 0.8 and avg_logprob < -0.5:
        return True

    # A stock phrase is only evidence when the segment is already doubtful.
    return stripped in HALLUCINATED_ON_SILENCE and (
        no_speech_prob > 0.5 or avg_logprob < -0.7
    )


def resample_to_16k(pcm: bytes, source_rate: int) -> np.ndarray:
    """
    Convert int16 PCM to the float32 mono 16kHz Whisper expects.

    Linear interpolation, no anti-aliasing filter. That introduces aliasing
    when downsampling, which Whisper tolerates well and a proper resampler
    would avoid. Revisit if transcript quality is ever traced back to here.
    """
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if source_rate == WHISPER_SAMPLE_RATE or samples.size == 0:
        return samples
    ratio = WHISPER_SAMPLE_RATE / source_rate
    target_len = round(samples.size * ratio)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, samples.size - 1, target_len, dtype=np.float32)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


class WhisperSttAdapter(SttAdapter):
    """Sliding window Whisper, emitting cumulative hypotheses on a timer."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "tiny",
        language: str = "en",
        device: str = "cpu",
        compute_type: str | None = None,
        emit_interval: float = 0.5,
        max_window_s: float = 8.0,
        silence_rms: float = 0.005,
        silence_duration_s: float = 0.7,
        speech_rms: float = 0.006,
        vad_threshold: float = 0.5,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.emit_interval = emit_interval
        """Wall seconds between transcription passes. Lower costs more CPU."""

        self.max_window_s = max_window_s
        """
        Hard cap on the buffer, and a latency source in its own right.

        This buffer sits INSIDE the adapter, so session/backpressure.py cannot
        see it. Guarding the room-to-STT queue at 3s while this held 20s still
        gave a p50 of 13.8s, because the two queues are in series and only one
        was bounded. Whichever STT adapter is in use, its internal buffering
        counts against the same budget as everything else.

        Whisper's own window is 30 seconds. Going near it makes each pass
        slower with no accuracy gain for live speech; 8 keeps enough context
        for sentence-level punctuation without parking 20 seconds of audio.
        """

        self.silence_rms = silence_rms
        self.silence_duration_s = silence_duration_s

        self.speech_rms = speech_rms
        """
        Refuse to transcribe a buffer quieter than this.

        The cheapest defence against inventing words: a model asked about
        silence cannot answer with a stock phrase if it is never asked. It also
        saves the pass entirely, which on a machine that sheds audio is not
        incidental.
        """

        self.vad_threshold = vad_threshold

        self._model = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_audio = 0.0
        self._locked_text = ""
        """Transcript of audio already flushed out of the buffer. Hypotheses are
        cumulative from session start, which is what the chunker expects."""

        self._seq = 0
        self._quiet_for = 0.0
        self._over_budget_passes = 0
        self._warned_unsustainable = False
        self._suppressed = 0

    # -------------------------------------------------------------------------

    @property
    def required_sample_rate(self) -> int:
        return WHISPER_SAMPLE_RATE

    @property
    def emits_interims(self) -> bool:
        return True

    @property
    def latency_credible(self) -> bool:
        """Always False. See the module docstring."""
        return False

    async def __aenter__(self) -> Self:
        from faster_whisper import WhisperModel

        started = time.monotonic()
        self._model = await asyncio.to_thread(
            WhisperModel, self.model_size, device=self.device, compute_type=self.compute_type
        )
        log.warning(
            "Whisper STT loaded: development only, latency figures are not credible",
            extra={
                "model": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "load_s": round(time.monotonic() - started, 2),
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._model = None

    # -------------------------------------------------------------------------

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        if self._model is None:
            raise RuntimeError("use WhisperSttAdapter as an async context manager")

        next_emit = 0.0
        async for frame in frames:
            audio = resample_to_16k(frame.pcm, frame.sample_rate)
            if self._buffer.size == 0:
                self._buffer_start_audio = frame.t_audio
            self._buffer = np.concatenate([self._buffer, audio])
            self._track_silence(audio, frame.duration)

            window_s = self._buffer.size / WHISPER_SAMPLE_RATE
            quiet = self._quiet_for >= self.silence_duration_s
            overlong = window_s >= self.max_window_s

            if frame.t_wall < next_emit and not quiet and not overlong:
                continue
            next_emit = frame.t_wall + self.emit_interval

            text = await self._transcribe_buffer()
            if not text and not quiet:
                continue

            # Silence and the window cap both close the current utterance: its
            # text moves into the locked prefix and the buffer is cleared.
            finalise = quiet or overlong
            yield self._hypothesis(text, frame, is_final=finalise)

            if finalise:
                self._locked_text = f"{self._locked_text} {text}".strip()
                self._buffer = np.zeros(0, dtype=np.float32)
                self._quiet_for = 0.0

        if self._buffer.size:
            text = await self._transcribe_buffer()
            if text:
                yield self._hypothesis(
                    text,
                    AudioFrame(b"", WHISPER_SAMPLE_RATE, self._current_audio_end(), time.monotonic()),
                    is_final=True,
                )

    # -------------------------------------------------------------------------

    def _track_silence(self, audio: np.ndarray, duration: float) -> None:
        if audio.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(audio))))
        self._quiet_for = self._quiet_for + duration if rms < self.silence_rms else 0.0

    def _current_audio_end(self) -> float:
        return self._buffer_start_audio + self._buffer.size / WHISPER_SAMPLE_RATE

    async def _transcribe_buffer(self) -> str:
        if self._buffer.size == 0:
            return ""
        buffer = self._buffer.copy()

        # Do not ask the model about silence.
        level = float(np.sqrt(np.mean(np.square(buffer))))
        if level < self.speech_rms:
            return ""

        started = time.monotonic()

        def run() -> str:
            from faster_whisper.vad import VadOptions

            segments, _info = self._model.transcribe(
                buffer,
                language=self.language,
                beam_size=1,
                # Never carry context across passes: a hallucination in one
                # window otherwise seeds the next.
                condition_on_previous_text=False,
                vad_filter=True,
                # Explicit rather than default, so an upstream change cannot
                # quietly loosen the gate that keeps room tone out.
                vad_parameters=VadOptions(
                    threshold=self.vad_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=400,
                    speech_pad_ms=200,
                ),
                # Whisper's own guards, tightened. A segment the model itself
                # doubts is not worth translating.
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            kept = []
            for seg in segments:
                if is_hallucination(seg.text, seg.no_speech_prob, seg.avg_logprob):
                    self._suppressed += 1
                    log.debug(
                        "suppressed a likely hallucination",
                        extra={
                            "text": seg.text.strip(),
                            "no_speech_prob": round(seg.no_speech_prob, 3),
                            "avg_logprob": round(seg.avg_logprob, 3),
                        },
                    )
                    continue
                kept.append(seg.text.strip())
            return " ".join(kept).strip()

        text = await asyncio.to_thread(run)
        self._note_pass_cost(time.monotonic() - started, buffer.size / WHISPER_SAMPLE_RATE)
        return text

    def _note_pass_cost(self, elapsed_s: float, window_s: float) -> None:
        """
        Warn when the configuration cannot keep up, and say what to change.

        A sliding-window backend re-transcribes its whole buffer every pass, so
        one pass costs about `window_s * RTF` while advancing only
        `emit_interval` of new audio. Sustainable requires:

            emit_interval  >  window_s * RTF

        Judged on consecutive passes, not on an average. The buffer starts
        empty and grows, so early passes are cheap; averaging them in hides the
        steady-state cost, which is the whole point. An earlier version
        averaged the last three and stayed silent through a run that shed 23%
        of the audio.

        A streaming transducer has no such constraint: it consumes each frame
        once and never revisits it.
        """
        if window_s <= 0:
            return

        if elapsed_s <= self.emit_interval:
            self._over_budget_passes = 0
            return

        self._over_budget_passes += 1
        if self._over_budget_passes < 2 or self._warned_unsustainable:
            return

        self._warned_unsustainable = True

        # Pass cost is WALL time, which is the right measure: what decides
        # whether we keep up is elapsed time, not pure inference time. But it
        # therefore includes CPU contention with TTS and translation, so
        # dividing it by the window does not give a real-time factor and must
        # not be reported as one. A near-empty window makes that ratio explode.
        extra = {
            "pass_cost_s": round(elapsed_s, 2),
            "emit_interval_s": self.emit_interval,
            "window_s": round(window_s, 1),
            "raise_emit_interval_to_s": round(elapsed_s * 1.2, 1),
        }
        if window_s >= 1.0:
            ratio = elapsed_s / window_s
            extra["cost_per_audio_second"] = round(ratio, 3)
            extra["or_lower_window_to_s"] = round(self.emit_interval / ratio, 1)
        else:
            # Costing this much on an almost empty buffer means the time went
            # somewhere other than transcription: thread starvation or an
            # event loop that could not get back to us. A smaller model is the
            # fix, not a smaller window.
            extra["note"] = (
                "pass was expensive on a near-empty buffer, so this is CPU "
                "contention rather than transcription cost; use a smaller model"
            )
        log.error(
            "STT cannot keep up: a pass costs more than the emit interval, so "
            "the backlog will grow without bound and audio will be dropped",
            extra=extra,
        )

    def _hypothesis(self, window_text: str, frame: AudioFrame, is_final: bool) -> Hypothesis:
        self._seq += 1
        return Hypothesis(
            text=f"{self._locked_text} {window_text}".strip(),
            is_final=is_final,
            t_audio_start=0.0,
            t_audio_end=self._current_audio_end(),
            t_wall=time.monotonic(),
            seq=self._seq,
            language=self.language,
        )
