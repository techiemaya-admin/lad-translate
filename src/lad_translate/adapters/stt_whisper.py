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
import re
import time
from collections import Counter
from collections.abc import AsyncIterator
from typing import Self

import numpy as np

from ..obs.log import get_logger
from .audio import SAMPLE_RATE_16K, resample_to_16k
from .base import AudioFrame, Hypothesis, SttAdapter

log = get_logger(__name__)

WHISPER_SAMPLE_RATE = SAMPLE_RATE_16K

# Phrases Whisper emits when there is nothing to transcribe. They come from its
# training data, which is full of YouTube audio, and they appear over silence
# and room tone with high confidence in the text itself.
#
# Observed live: "Thanks for watching!" and a long run of bare "Thank you."
# in a session where nobody said either. Translated and spoken to an audience,
# an invented politeness is worse than a gap.
#
# A later session on a real phone got four more past the gate, because the match
# was exact against the raw string and these did not appear in it verbatim:
#
#     "I'll see you later."
#     "I hope you enjoyed this video. Thanks."
#     "And I hope you enjoyed this video. I hope you enjoyed this video."
#     "I'll see you next time."
#
# All four were translated and spoken to the room. The list is stored normalised
# now, and the segment is normalised the same way before it is looked up, so a
# trailing full stop or an exclamation mark no longer needs its own entry.
#
# Matched only when the segment ALSO looks like non-speech. "Thank you" is a
# real thing people say at a conference, and a blanket blocklist would delete
# genuine speech to remove an artefact.
_STOCK_PHRASES = (
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thanks for watching and see you next time", "see you next time",
    "i'll see you next time", "see you later", "i'll see you later",
    "you", "bye", "okay",
)

# The distinction the gate rests on is whether the words have an innocent
# reading in THIS room. "Thank you" does, so it is only evidence when the
# segment already looks doubtful. A reference to a video, an episode or a
# channel does not: the speaker is addressing a hall through an interpreter,
# not signing off a broadcast. Those are dropped on the words alone.
#
# Reported live, and the reason this tier exists: a phone heard
# "This is the end of the day, and I will meet you in the next episode."
# in the middle of someone reading a device manual aloud. It is not on any list
# and never will be -- Whisper paraphrases these freely -- so matching the SHAPE
# is the only thing that keeps up.
_BROADCAST_NOUNS = r"(?:video|episode|channel|stream|livestream|vlog|podcast|tutorial)"
_FAREWELL = r"(?:see|meet|catch|talk to)\s+(?:you|ya)|until\s+next|till\s+next|goodbye"

# A farewell aimed at a broadcast audience. Both halves are required, so
# "let's watch the video" and "I'll see you at lunch" both survive.
_BROADCAST_OUTRO = re.compile(
    rf"(?:{_FAREWELL})[^.!?]*\b{_BROADCAST_NOUNS}\b"
    rf"|\b{_BROADCAST_NOUNS}\b[^.!?]*(?:{_FAREWELL})",
    re.IGNORECASE,
)

# Inviting the audience to act on a video: subscribing, liking, hitting a bell,
# enjoying it. Same reasoning -- no live speaker says these to a room.
_BROADCAST_CALL = re.compile(
    rf"\b(?:subscribe|like and subscribe|hit the bell)\b"
    rf"|\b(?:hope|hoped)\s+you\s+(?:enjoyed|liked|enjoy)\b[^.!?]*\b{_BROADCAST_NOUNS}\b"
    rf"|\bthanks?\s+(?:you\s+)?for\s+watching\b"
    rf"|\bsubtitles?\s+by\b",
    re.IGNORECASE,
)


def is_broadcast_artefact(text: str) -> bool:
    """
    Whether the words belong to a video sign-off rather than to this room.

    Checked on the words alone, with no confidence gate, because unlike "thank
    you" there is no reading of these that a live interpretation session should
    ever carry to an audience.
    """
    return bool(_BROADCAST_OUTRO.search(text) or _BROADCAST_CALL.search(text))


# Clause punctuation, not just sentence punctuation. Whisper's loops are as
# often comma-separated as full-stopped, and splitting only on ".!?" reads
# "I'm Cassie, I'm Cassie, I'm Cassie." as a single sentence with nothing
# repeated in it.
_CLAUSE_SPLIT = re.compile(r"[.!?,;:]+")

LOG_PROB_FLOOR = -1.0
"""Below this the model was guessing at the tokens, so the words mean nothing.

The same value is passed to Whisper as `log_prob_threshold`, where it governs
temperature fallback rather than whether the segment is used. Enforcing it here
is what makes it a floor.
"""

LOOP_MIN_UNITS = 3
LOOP_MIN_REPEATS = 3
LOOP_MIN_WORDS = 2
LOOP_SHARE = 0.6


def _clauses(text: str) -> list[str]:
    return [c for c in (_normalise(p) for p in _CLAUSE_SPLIT.split(text)) if c]


def is_repetition_loop(text: str) -> bool:
    """
    Whether the decoder got stuck repeating one phrase.

    Observed live, all four in one session, and none catchable by a phrase list:

        I'm Cassie, I'm Cassie, I'm Cassie.
        I guess you'll. I guess you'll. I guess you'll.
        I love you. I love you, I love you. I love you, I love you.
        Mehtun Sibya Arkanda. Mehtun Sibya Arkanda. Mehtun Sibya Arkanda. Mehtun.

    The shape is one clause occupying most of the segment. Requiring it to fill
    at least LOOP_SHARE of the clauses is what separates a loop from ordinary
    speech that happens to repeat a phrase: "Hello, how are you? I'm quick, how
    are you?" repeats "how are you" twice out of four clauses and is kept.

    Single words are exempt however often they repeat, because "No, no, no" and
    "Yes, yes, yes" are things people say and mean.

    Ungated, like the broadcast sign-offs. `compression_ratio_threshold` is
    meant to catch this inside the model and demonstrably did not, and these
    reached an audience while every confidence signal was happy.
    """
    units = _clauses(text)
    if len(units) < LOOP_MIN_UNITS:
        return False
    unit, count = Counter(units).most_common(1)[0]
    if count < LOOP_MIN_REPEATS or len(unit.split()) < LOOP_MIN_WORDS:
        return False
    return count / len(units) >= LOOP_SHARE

# Leading discourse particles. Whisper prefixes its stock phrases with these
# often enough that "And I hope you enjoyed this video" would otherwise miss a
# list that holds the phrase itself. Stripping one only ever exposes the rest of
# the segment to the same whole-segment match; it never turns a real sentence
# into a stock one, because what remains still has to match in full.
_LEADING_FILLER = ("and", "so", "but", "well", "okay", "ok", "um", "uh", "oh")


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    cleaned = cleaned.strip(".,!?;:-\"'“”‘’ ")
    words = cleaned.split(" ")
    while len(words) > 1 and words[0] in _LEADING_FILLER:
        words = words[1:]
    return " ".join(words).strip()


HALLUCINATED_ON_SILENCE = frozenset(_normalise(p) for p in _STOCK_PHRASES)


def _sentences(text: str) -> list[str]:
    return [s for s in (_normalise(p) for p in re.split(r"[.!?]+", text)) if s]


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

    Matching stays whole-segment. A substring test would delete "Thank you all
    for coming to the summit today", which is a real thing a chair says, so a
    segment of several sentences is matched by requiring EVERY sentence in it to
    be stock rather than by looking for one anywhere inside.

    Broadcast sign-offs are the exception and are dropped on the words alone.
    The gate above exists because "thank you" is ambiguous; "I will meet you in
    the next episode" is not, and it reached a live audience precisely because
    the model was confident enough to clear every threshold here.
    """
    stripped = text.strip()
    if not _normalise(stripped):
        return True

    # Confidently non-speech: drop whatever it produced, whatever the words.
    if no_speech_prob > 0.8 and avg_logprob < -0.5:
        return True

    # Below the model's own confidence floor: it was guessing at the tokens.
    #
    # Measured in a live session where the speaker was silent, with every kept
    # segment's signals logged. `no_speech_prob` was USELESS -- 0.013 to 0.318
    # on invented text, so the branch above never fired -- while `avg_logprob`
    # separated the two cleanly enough to act on:
    #
    #     -1.431  Me too?              -0.877  Hello.
    #     -1.331  I'll explain it.     -0.752  Bye bye.
    #     -1.284  Matthew.             -0.632  or you do it.
    #     -1.207  or you'll be sick.   -0.630  and all good morning.
    #     -1.098  Alvin Dab.           -0.594  At the main.
    #     -1.014  more kebab           -0.580  I'll think that.
    #
    # The threshold is not a new invention: LOG_PROB_FLOOR is the same -1.0 this
    # adapter already hands Whisper as `log_prob_threshold`, which the model uses
    # to decide whether to retry at a higher temperature but never to discard.
    # The floor was already declared; nothing was enforcing it.
    #
    # This does discard badly transcribed real speech as well. On a backend
    # measured at 14.8% WER streaming, a segment the model itself rates this
    # poorly was unlikely to survive translation intact, and a gap is better
    # than a confident invention.
    if avg_logprob < LOG_PROB_FLOOR:
        return True

    # A video sign-off is wrong in this room however confident the model is,
    # and confidence is exactly what the earlier gate could not rely on: these
    # reached a live audience while the model was sure enough to pass every
    # threshold above. A decoder loop is the same case.
    if is_broadcast_artefact(stripped) or is_repetition_loop(stripped):
        return True

    # Everything below is evidence only once the segment is already doubtful.
    if not (no_speech_prob > 0.5 or avg_logprob < -0.7):
        return False

    if _normalise(stripped) in HALLUCINATED_ON_SILENCE:
        return True

    sentences = _sentences(stripped)
    if sentences and all(s in HALLUCINATED_ON_SILENCE for s in sentences):
        return True

    # Whisper loops when it has nothing to work with, repeating one phrase for
    # the whole window. That shape is a hallucination whatever the words are,
    # which is what makes it worth testing separately from the list: it catches
    # the loops nobody has seen yet. Short interjections are excluded because
    # "no, no, no" and "yes, yes" are things people really say.
    return (
        len(sentences) > 1
        and len(set(sentences)) == 1
        and len(sentences[0].split()) >= 3
    )


class WhisperSttAdapter(SttAdapter):
    """Sliding window Whisper, emitting cumulative hypotheses on a timer."""

    name = "faster-whisper"

    DEFAULT_SPEECH_RMS = 0.006
    """Suits a recorded fixture, and is too low for a microphone in a room.

    Measured, 6s buffers, float32:

        fixtures/holmes.wav        0.029 - 0.050    quiet narration
        fixtures/jfk.wav           0.059 - 0.113
        a phone, speaker silent    0.0002           true silence
        a phone, background noise  0.008 - 0.037    movement, distant voices
        a phone, speaker talking   0.067 - 0.125

    There is no single value that serves both. A gate above the phone's noise
    band would silence holmes.wav, which is the fixture the WER numbers are
    scored against; a gate below it hands the model noise to invent words from.
    One room's noise floor is another recording's speech, so this is a
    deployment setting rather than a constant, and the library default stays
    where the fixtures need it. The tools that drive a live microphone default
    to LIVE_SPEECH_RMS instead.
    """

    LIVE_SPEECH_RMS = 0.05
    """What the tools use, and what a live microphone in a room needs.

    Sits in the empty band between the phone's background noise, which topped
    out at 0.037, and its speech, which started at 0.067. Measured in one room
    on one handset: it is a better starting point than 0.006 for a live mic and
    it is still a starting point. Confirm it per venue from the `transcribing
    buffer` and `buffer below the speech gate` lines at DEBUG.

    Too high and a quiet speaker goes silent with nothing in the transcript to
    explain it, which is the failure this value can cause and the reason every
    tool that applies it also exposes a flag to lower it.
    """

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
        speech_rms: float = DEFAULT_SPEECH_RMS,
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
        #
        # The level is logged either way. This gate is an absolute threshold on
        # a signal whose floor depends on the microphone and the room, so the
        # only way to know whether it is set right for a given venue is to see
        # what the buffers actually measured there.
        level = float(np.sqrt(np.mean(np.square(buffer))))
        if level < self.speech_rms:
            log.debug(
                "buffer below the speech gate, not transcribed",
                extra={"rms": round(level, 5), "gate": self.speech_rms},
            )
            return ""
        log.debug(
            "transcribing buffer",
            extra={"rms": round(level, 5), "gate": self.speech_rms},
        )

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
                log_prob_threshold=LOG_PROB_FLOOR,
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
                # Kept segments carry their signals too. Without this there is
                # no way to tell, after the fact, whether something that reached
                # the audience slipped past a gate or was never doubted at all,
                # and that is the first question asked every time one does.
                log.debug(
                    "kept a segment",
                    extra={
                        "text": seg.text.strip(),
                        "no_speech_prob": round(seg.no_speech_prob, 3),
                        "avg_logprob": round(seg.avg_logprob, 3),
                    },
                )
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
