"""
NVIDIA cache-aware streaming FastConformer. THE PRODUCTION STT. NOT YET RUN.

Written against NeMo's documented streaming API and verified line by line
against the reference implementation in
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py,
but never executed: NeMo needs torch with CUDA, and the development machine is
a two core Intel Mac. Treat the tensor path as unverified until it has run on
the A4000. The frame arithmetic, which is where this kind of adapter usually
goes wrong, is pure and IS tested here -- see ChunkSchedule.

WHY THIS ONE MATTERS MORE THAN THE OTHERS

Whisper is a 30 second window model. Every streaming wrapper around it, this
project's included, re-transcribes a sliding buffer, which is where the
constraint

    emit_interval > window_s * RTF

comes from and why Whisper small shed 74% of the audio on two cores. Faster
hardware loosens that; it does not remove it, because the window is
architectural.

A cache-aware streaming transducer removes it. Each step encodes only the new
audio and carries its left context in a cache tensor, so the cost per step is
constant no matter how long the speaker has been talking. Twenty minutes in
costs exactly what the first second cost. That is the property this product
needs, and it is why every measurement in this repo has ended at the same
conclusion: STT is the sole remaining ceiling.

THE LOOKAHEAD IS THE LATENCY KNOB, AND IT IS SPENT BEFORE WE DO ANYTHING

This model was trained with four lookaheads in one set of weights, selectable
at load time with no retraining. The lookahead is time the model waits for
future audio before committing, so it is subtracted from the latency budget
before translation or synthesis has begun:

    att_context_size   lookahead   WER (RNNT, LS test-other)
    [70, 0]                0 ms    7.0%
    [70, 1]               80 ms    6.4%
    [70, 6]              480 ms    5.7%
    [70, 13]            1040 ms    5.4%

NeMo defaults to [70, 13]. This adapter defaults to [70, 6], because 1040ms is
more than half a 2s budget spent before the first word reaches the translator,
and the 0.3 WER points it buys back do not pay for it. That is a product
decision, made here explicitly rather than inherited from a library default.

WHAT IT DOES NOT DO

    English only       "multi" in the model name means multiple lookaheads,
                       NOT multilingual. Easy to misread. A non-English
                       speaker needs a different model entirely.
    114M params        small enough to share the A4000 with NLLB and the TTS
                       voices, which Qwen3-ASR at 1.7B is not.

WHAT IT DOES THAT THE OTHER TWO CANNOT

It gives us honest word timings. Whisper's streaming path times words badly
and Qwen3-ASR's streaming mode returns none at all, so both make the chunker
interpolate across a hypothesis span and every latency figure inherits that
error. Here each word is stamped with the audio position of the encoder step
that first produced it, which is an upper bound accurate to one shift (about
560ms at the default lookahead) and, unlike interpolation, never wrong in the
optimistic direction. See WordClock.

ONE INTERACTION WITH THE CHUNKER, WORTH KNOWING BEFORE TUNING

An RNNT hypothesis threaded through previous_hypotheses is append only: the
model does not take words back. The chunker's LocalAgreement-n exists to find
stability in output that DOES get revised, so against this backend the
agreement window costs a step of latency per unit of n and buys nothing. Steps
land roughly every 560ms, so agreement_n=2 is about 560ms of pure delay.

    ChunkerConfig(agreement_n=1)

is the right setting here, and `revises_hypotheses` on this class says so
programmatically. The CTC decoder is a different matter: it re-decodes the
whole prefix each step and genuinely can revise, so it should keep n=2.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from ..obs.log import get_logger
from .audio import SAMPLE_RATE_16K, resample_to_16k
from .base import AudioFrame, Hypothesis, SttAdapter, WordTiming

log = get_logger(__name__)

DEFAULT_MODEL = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"

WINDOW_STRIDE_S = 0.01
"""Seconds of audio per feature frame. The model's preprocessor is configured
with window_stride=0.01, and every frame count in this module converts to
seconds through it."""


@dataclass(frozen=True, slots=True)
class Lookahead:
    """One of the model's trained lookahead settings."""

    att_context_size: tuple[int, int]
    ms: int
    wer_rnnt: float
    wer_ctc: float


LOOKAHEADS: dict[str, Lookahead] = {
    # ms = att_context_size[1] * subsampling_factor(8) * window_stride(0.01)
    "0ms": Lookahead((70, 0), 0, 7.0, 8.4),
    "80ms": Lookahead((70, 1), 80, 6.4, 7.8),
    "480ms": Lookahead((70, 6), 480, 5.7, 6.7),
    "1040ms": Lookahead((70, 13), 1040, 5.4, 6.2),
}

DEFAULT_LOOKAHEAD = "480ms"
"""Not NeMo's 1040ms default. See the module docstring: on a 2s glass to glass
budget, 1040ms of it spent before translation starts is not affordable, and
480ms costs 0.3 WER points to get 560ms back."""


def lookahead_ms(att_context_size: tuple[int, int], subsampling_factor: int = 8) -> float:
    """
    Milliseconds of future audio a given attention context waits for.

    This is the formula NeMo documents, reproduced so the table above can be
    checked rather than trusted.
    """
    return att_context_size[1] * subsampling_factor * WINDOW_STRIDE_S * 1000


# =============================================================================
# FRAME ARITHMETIC
#
# Pure integers, no torch. This is the part that is easy to get subtly wrong
# and impossible to notice: an off-by-one in the pre-encode cache does not
# raise, it just quietly degrades the transcript. So it lives here, where the
# development machine can test it.
# =============================================================================


@dataclass(frozen=True, slots=True)
class StreamingGeometry:
    """
    The encoder's streaming_cfg, normalised.

    NeMo stores several of these as either an int or a two element list, where
    element 0 applies to the first step and element 1 to every step after. That
    branching appears four times in its buffer implementation; doing it once
    here is the whole point of this class.
    """

    chunk_size_first: int
    chunk_size: int
    shift_size_first: int
    shift_size: int
    pre_encode_cache_first: int
    pre_encode_cache: int
    drop_extra_pre_encoded: int
    subsampling_factor: int = 8

    @staticmethod
    def _pair(value: Any) -> tuple[int, int]:
        """Normalise int-or-list into (first_step, steady_state)."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"expected a 2 element streaming_cfg value, got {value!r}")
            return int(value[0]), int(value[1])
        return int(value), int(value)

    @classmethod
    def from_encoder(cls, encoder: Any) -> StreamingGeometry:
        cfg = encoder.streaming_cfg
        chunk_first, chunk = cls._pair(cfg.chunk_size)
        shift_first, shift = cls._pair(cfg.shift_size)
        cache_first, cache = cls._pair(cfg.pre_encode_cache_size)
        return cls(
            chunk_size_first=chunk_first,
            chunk_size=chunk,
            shift_size_first=shift_first,
            shift_size=shift,
            pre_encode_cache_first=cache_first,
            pre_encode_cache=cache,
            drop_extra_pre_encoded=int(cfg.drop_extra_pre_encoded),
            subsampling_factor=int(getattr(encoder, "subsampling_factor", 8)),
        )

    @property
    def step_interval_s(self) -> float:
        """Audio consumed per steady state step. The hypothesis emit cadence."""
        return self.shift_size * WINDOW_STRIDE_S

    def audio_time(self, frame: int) -> float:
        """Seconds of source audio up to a feature frame index."""
        return frame * WINDOW_STRIDE_S


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """
    One encoder step, described in absolute feature frame indices.

    Absolute means "counted from the first frame of the session", not from the
    start of whatever the buffer currently holds. Keeping the plan absolute is
    what lets the buffer discard consumed frames without the schedule having to
    know it happened.
    """

    step: int
    start: int
    """First frame of the chunk proper."""

    end: int
    """One past the last frame of the chunk."""

    cache_start: int
    """First real history frame prepended to the chunk."""

    zero_pad: int
    """
    History frames that do not exist yet and must be supplied as zeros.

    Non-zero only near the start of a session, where there is not yet enough
    history to fill the pre-encode cache.
    """

    drop_extra_pre_encoded: int
    """
    Encoder outputs to discard from the front, because they correspond to the
    prepended history rather than to new audio. Zero on the first step, where
    nothing real was prepended.
    """

    @property
    def cache_frames(self) -> int:
        return self.start - self.cache_start

    @property
    def width(self) -> int:
        """Total frames handed to the encoder, history and padding included."""
        return self.zero_pad + self.cache_frames + (self.end - self.start)


class ChunkSchedule:
    """
    Decides which frames form the next encoder step, and which may be dropped.

    NeMo's CacheAwareStreamingAudioBuffer does this job, and this class exists
    because that one cannot be used for a live session. It is a file simulator:
    it pads and rewrites its whole buffer on every append, which is O(n) per
    append and O(n^2) over a session, it never releases consumed audio, and its
    iterator RETURNS when the buffer runs dry rather than waiting for more. On a
    ninety minute keynote the first two are hundreds of megabytes of features
    held for no reason, and the third means the loop exits the moment the
    speaker pauses long enough for the queue to empty.

    So the chunking is reimplemented here, following the same arithmetic, with
    two changes: it yields only when a whole chunk is genuinely available, and
    it reports `retain_from` so the caller can discard everything older.
    """

    def __init__(self, geometry: StreamingGeometry) -> None:
        self.geometry = geometry
        self.step = 0
        self.cursor = 0
        """Absolute frame index where the next chunk begins."""

    def _sizes(self) -> tuple[int, int, int, int]:
        """(chunk, shift, pre_encode_cache, drop_extra) for the current step."""
        g = self.geometry
        if self.step == 0:
            # No caching has happened yet, so there is nothing to drop. This
            # mirrors NeMo's calc_drop_extra_pre_encoded, which returns 0 for
            # step 0 for exactly that reason.
            return g.chunk_size_first, g.shift_size_first, g.pre_encode_cache_first, 0
        return g.chunk_size, g.shift_size, g.pre_encode_cache, g.drop_extra_pre_encoded

    def offer(self, available: int) -> list[ChunkPlan]:
        """
        Given the total frames received so far, return the steps now runnable.

        `available` is an absolute count from session start and only ever
        grows. A partial chunk yields nothing: the encoder is entitled to a
        full chunk, and handing it a short one produces output for audio that
        has not arrived.
        """
        plans: list[ChunkPlan] = []
        while True:
            chunk, shift, cache, drop = self._sizes()
            end = self.cursor + chunk
            if end > available:
                return plans
            cache_frames = min(cache, self.cursor)
            plans.append(
                ChunkPlan(
                    step=self.step,
                    start=self.cursor,
                    end=end,
                    cache_start=self.cursor - cache_frames,
                    zero_pad=cache - cache_frames,
                    drop_extra_pre_encoded=drop,
                )
            )
            self.cursor += shift
            self.step += 1

    def flush(self, available: int) -> ChunkPlan | None:
        """
        The final, short step at end of session.

        Everything left over goes through in one pass with whatever width it
        has, because there is no more audio coming and the alternative is
        silently dropping the speaker's last few words. Only for session end:
        during a session a short chunk means "wait", not "flush".
        """
        if available <= self.cursor:
            return None
        _, _, cache, drop = self._sizes()
        cache_frames = min(cache, self.cursor)
        plan = ChunkPlan(
            step=self.step,
            start=self.cursor,
            end=available,
            cache_start=self.cursor - cache_frames,
            zero_pad=cache - cache_frames,
            drop_extra_pre_encoded=drop,
        )
        self.cursor = available
        self.step += 1
        return plan

    @property
    def retain_from(self) -> int:
        """
        Oldest frame the next step could still need. Everything before it goes.

        This is the bound that makes a multi hour session possible: memory
        holds one chunk plus one pre-encode cache, not the whole talk.
        """
        _, _, cache, _ = self._sizes()
        return max(0, self.cursor - cache)


class FrameWindow:
    """
    Absolute-to-local index bookkeeping for a buffer that discards its head.

    Separated from the tensor it describes for one reason: this arithmetic is
    where a streaming adapter silently goes wrong, and an off-by-one in it does
    not raise, it just feeds the encoder the wrong audio. Torch is not
    installed on the development machine, so anything that touches a tensor
    cannot be tested until the A4000 is available. This can be, and is.
    """

    __slots__ = ("origin", "total")

    def __init__(self) -> None:
        self.origin = 0
        """Absolute index of the frame currently at local position 0."""

        self.total = 0
        """Absolute count of frames ever appended."""

    @property
    def held(self) -> int:
        """Frames still in memory. The number that must not grow without bound."""
        return self.total - self.origin

    def extend(self, count: int) -> None:
        if count < 0:
            raise ValueError(f"cannot extend by {count} frames")
        self.total += count

    def local(self, start: int, end: int) -> tuple[int, int]:
        """Translate an absolute frame range into offsets into what is held."""
        if start < self.origin:
            raise IndexError(
                f"frame {start} was already discarded (window starts at {self.origin}); "
                "the schedule and the buffer have diverged"
            )
        if end > self.total:
            raise IndexError(
                f"frame {end} has not arrived yet (only {self.total} received); "
                "a step was scheduled against audio that does not exist"
            )
        return start - self.origin, end - self.origin

    def drop_before(self, absolute: int) -> int:
        """Discard everything older. Returns how many frames went."""
        target = min(max(absolute, self.origin), self.total)
        dropped = target - self.origin
        self.origin = target
        return dropped


class FeatureBuffer:
    """
    A bounded window over the feature stream, addressed in absolute frames.

    Callers append and slice using absolute indices; the window underneath
    tracks how much has already been thrown away. Slicing something discarded
    is a bug in the schedule, so it raises rather than quietly returning short.
    """

    def __init__(self) -> None:
        self._frames: Any = None
        """(1, features, time) tensor, or None before the first append."""

        self.window = FrameWindow()

    @property
    def available(self) -> int:
        return self.window.total

    def append(self, features: Any) -> None:
        import torch

        if self._frames is None:
            self._frames = features
        else:
            self._frames = torch.cat((self._frames, features), dim=-1)
        self.window.extend(int(features.size(-1)))

    def slice(self, start: int, end: int) -> Any:
        lo, hi = self.window.local(start, end)
        return self._frames[:, :, lo:hi]

    def discard_before(self, absolute: int) -> None:
        dropped = self.window.drop_before(absolute)
        if dropped and self._frames is not None:
            self._frames = self._frames[:, :, dropped:]


class WordClock:
    """
    Assigns each transcript word the audio position of the step that produced it.

    Deliberately NOT derived from the model's own token timestamps. Those
    exist, but under previous_hypotheses threading it is undocumented whether
    they are offset per step or relative to it, and a latency figure resting on
    a guess about that is worse than one resting on something plainly
    conservative. What is used instead is a fact the adapter knows for certain:
    a word that first appeared at step k came from audio ending at step k's
    boundary.

    That over-estimates by at most one shift, so latency built on it is a
    ceiling rather than an optimistic guess. On a product whose entire claim is
    that its numbers are measured, that is the correct direction to be wrong
    in. Once the token timestamps have been checked against real audio on the
    A4000 they can replace this and tighten it.

    RNNT output is append only, so in practice stamps are assigned once and
    never move. CTC re-decodes its whole prefix each step and genuinely can
    revise earlier words; when it does, the divergent tail is re-stamped at the
    current step rather than keeping a timing that belonged to text which has
    since changed.
    """

    __slots__ = ("_ends", "_words")

    def __init__(self) -> None:
        self._words: list[str] = []
        self._ends: list[float] = []

    def stamp(self, text: str, audio_end: float) -> tuple[WordTiming, ...]:
        words = text.split()
        shared = 0
        for old, new in zip(self._words, words):
            if old != new:
                break
            shared += 1

        del self._words[shared:]
        del self._ends[shared:]
        for word in words[shared:]:
            self._words.append(word)
            self._ends.append(audio_end)

        timings = []
        previous = 0.0
        for word, end in zip(self._words, self._ends):
            timings.append(WordTiming(word=word, t_audio_start=previous, t_audio_end=end))
            previous = end
        return tuple(timings)


# =============================================================================
# ADAPTER
# =============================================================================


class FastConformerSttAdapter(SttAdapter):
    """Cache-aware streaming transducer. Constant cost per step, any duration."""

    name = "fastconformer"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        lookahead: str = DEFAULT_LOOKAHEAD,
        decoder: str = "rnnt",
        device: str = "cuda",
        language: str = "en",
        preprocess_block_s: float = 0.32,
    ) -> None:
        if lookahead not in LOOKAHEADS:
            raise KeyError(
                f"unknown lookahead {lookahead!r}; the model was trained with "
                f"{sorted(LOOKAHEADS)} and no others"
            )
        if decoder not in ("rnnt", "ctc"):
            raise ValueError(f"decoder must be 'rnnt' or 'ctc', not {decoder!r}")

        self.model_name = model
        self.lookahead = LOOKAHEADS[lookahead]
        self.lookahead_name = lookahead
        self.decoder = decoder
        self.device = device
        self.language = language

        self.preprocess_block_s = preprocess_block_s
        """
        Raw audio accumulated before running the mel front end.

        Each block is preprocessed independently, which is what NeMo's own live
        microphone path does, and it introduces a small discontinuity at block
        edges because the analysis window cannot span them. Smaller blocks mean
        more edges; larger ones add latency before a step can run. Worth an A/B
        against the file path on hardware, which tools/score_stt.py can drive.
        """

        # Fail at construction, not when the audience is already seated. The
        # model itself loads in __aenter__; checking the package merely exists
        # costs nothing and turns a mid-event failure into a config error.
        if importlib.util.find_spec("nemo") is None:
            raise RuntimeError(
                "nemo_toolkit is not installed. It needs torch, and this "
                "backend needs a CUDA device to meet its latency budget; "
                "there is no useful CPU path."
            )

        self._model: Any = None
        self._preprocessor: Any = None
        self._geometry: StreamingGeometry | None = None
        self._seq = 0
        self._clock = WordClock()

    # -------------------------------------------------------------------------

    @property
    def required_sample_rate(self) -> int:
        return SAMPLE_RATE_16K

    @property
    def emits_interims(self) -> bool:
        return True

    @property
    def revises_hypotheses(self) -> bool:
        """
        Whether committed words can still change.

        False for RNNT, whose threaded hypothesis is append only. The chunker's
        agreement window is a defence against revision, so against an append
        only backend it is latency with nothing bought. Set agreement_n=1.
        """
        return self.decoder == "ctc"

    async def __aenter__(self) -> Self:
        import torch
        from nemo.collections.asr.models import ASRModel

        started = time.monotonic()
        model = await asyncio.to_thread(
            ASRModel.from_pretrained, model_name=self.model_name, map_location=self.device
        )
        model.eval()

        if not hasattr(model.encoder, "set_default_att_context_size"):
            raise RuntimeError(
                f"{self.model_name} does not support selectable lookahead, so it is "
                "not one of the cache-aware streaming models this adapter targets"
            )
        model.encoder.set_default_att_context_size(
            att_context_size=list(self.lookahead.att_context_size)
        )
        model.encoder.setup_streaming_params()

        if self.decoder == "ctc" and hasattr(model, "change_decoding_strategy"):
            model.change_decoding_strategy(decoder_type="ctc")

        self._model = model
        self._preprocessor = self._build_preprocessor(model)
        self._geometry = StreamingGeometry.from_encoder(model.encoder)

        log.info(
            "FastConformer loaded",
            extra={
                "model": self.model_name,
                "lookahead": self.lookahead_name,
                "lookahead_ms": lookahead_ms(
                    self.lookahead.att_context_size, self._geometry.subsampling_factor
                ),
                "decoder": self.decoder,
                "device": self.device,
                "step_interval_s": round(self._geometry.step_interval_s, 3),
                "revises_hypotheses": self.revises_hypotheses,
                "load_s": round(time.monotonic() - started, 2),
                "torch_cuda": torch.cuda.is_available(),
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._model = None
        self._preprocessor = None
        self._geometry = None

    def _build_preprocessor(self, model: Any) -> Any:
        """
        A copy of the model's front end with dithering and padding disabled.

        Both are training-time behaviours. Dither adds noise, which makes the
        same audio produce different features on different passes and would
        make a streaming/offline comparison meaningless; pad_to rounds each
        call up to a multiple, which would insert silence at every block edge.
        NeMo's own buffer does exactly this, for the same two reasons.
        """
        import copy

        from omegaconf import OmegaConf

        cfg = copy.deepcopy(model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        return model.from_config_dict(cfg.preprocessor).to(self.device)

    # -------------------------------------------------------------------------

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        if self._model is None or self._geometry is None:
            raise RuntimeError("use FastConformerSttAdapter as an async context manager")

        state = _StreamState(self._model)
        buffer = FeatureBuffer()
        schedule = ChunkSchedule(self._geometry)
        samples_per_block = int(SAMPLE_RATE_16K * self.preprocess_block_s)
        pending = np.zeros(0, dtype=np.float32)
        text = ""
        last_plan: ChunkPlan | None = None
        """The most recent step, so the final hypothesis can be stamped with a
        real audio position rather than a zero one."""

        async for frame in frames:
            pending = np.concatenate([pending, resample_to_16k(frame.pcm, frame.sample_rate)])
            while pending.size >= samples_per_block:
                block, pending = pending[:samples_per_block], pending[samples_per_block:]
                buffer.append(await asyncio.to_thread(self._featurise, block))

            # offer() advances the cursor for every plan it returns, so
            # retain_from already describes the state AFTER the whole batch.
            # Discarding inside this loop would throw away history that a later
            # plan in the same batch still needs -- which happens whenever
            # audio arrives faster than one chunk at a time, as it does after
            # any reconnect. Drop once, when the batch is finished.
            plans = schedule.offer(buffer.available)
            for plan in plans:
                text = await asyncio.to_thread(self._run_step, state, buffer, plan)
                last_plan = plan
                if text and text != state.last_emitted:
                    state.last_emitted = text
                    yield self._hypothesis(text, plan, is_final=False)
            if plans:
                buffer.discard_before(schedule.retain_from)

        # End of stream. Push whatever is left through, short chunk and all,
        # rather than losing the tail of the speaker's last sentence.
        if pending.size:
            buffer.append(await asyncio.to_thread(self._featurise, pending))
        final_plan = schedule.flush(buffer.available)
        if final_plan is not None:
            text = await asyncio.to_thread(self._run_step, state, buffer, final_plan, True)
            last_plan = final_plan
        if text and last_plan is not None:
            yield self._hypothesis(text, last_plan, is_final=True)

    def _featurise(self, block: np.ndarray) -> Any:
        import torch

        signal = torch.from_numpy(block).unsqueeze(0).to(self.device)
        length = torch.tensor([block.shape[0]], device=self.device)
        with torch.inference_mode():
            features, _ = self._preprocessor(input_signal=signal, length=length)
        return features

    def _run_step(
        self,
        state: _StreamState,
        buffer: FeatureBuffer,
        plan: ChunkPlan,
        keep_all_outputs: bool = False,
    ) -> str:
        """
        One encoder step. Blocking; always called through asyncio.to_thread.

        keep_all_outputs is False during a session and True only on the final
        step: NeMo drops trailing outputs each step because the next step will
        recompute them with more context, which is right until there is no next
        step.
        """
        import torch

        chunk = buffer.slice(plan.cache_start, plan.end)
        if plan.zero_pad:
            pad = torch.zeros(
                (chunk.size(0), chunk.size(1), plan.zero_pad),
                device=chunk.device,
                dtype=chunk.dtype,
            )
            chunk = torch.cat((pad, chunk), dim=-1)
        lengths = torch.tensor([chunk.size(-1)], device=chunk.device)

        with torch.inference_mode():
            (
                state.pred_out,
                transcribed,
                state.cache_last_channel,
                state.cache_last_time,
                state.cache_last_channel_len,
                state.hypotheses,
            ) = self._model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=lengths,
                cache_last_channel=state.cache_last_channel,
                cache_last_time=state.cache_last_time,
                cache_last_channel_len=state.cache_last_channel_len,
                keep_all_outputs=keep_all_outputs,
                previous_hypotheses=state.hypotheses,
                previous_pred_out=state.pred_out,
                drop_extra_pre_encoded=plan.drop_extra_pre_encoded,
                return_transcription=True,
            )
        return _extract_text(transcribed)

    # -------------------------------------------------------------------------

    def _hypothesis(self, text: str, plan: ChunkPlan, is_final: bool) -> Hypothesis:
        assert self._geometry is not None
        self._seq += 1
        audio_end = self._geometry.audio_time(plan.end)
        return Hypothesis(
            text=text,
            is_final=is_final,
            t_audio_start=0.0,
            t_audio_end=audio_end,
            t_wall=time.monotonic(),
            seq=self._seq,
            language=self.language,
            words=self._clock.stamp(text, audio_end),
        )



class _StreamState:
    """
    Everything carried from one encoder step to the next.

    Not a dataclass: the cache tensors have to come from the encoder rather
    than from defaults, and a @dataclass would generate an __init__ that
    silently replaced this one, leaving every cache None.
    """

    __slots__ = (
        "cache_last_channel",
        "cache_last_channel_len",
        "cache_last_time",
        "hypotheses",
        "last_emitted",
        "pred_out",
    )

    def __init__(self, model: Any) -> None:
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)
        self.hypotheses = None
        self.pred_out = None
        self.last_emitted = ""


def _extract_text(transcribed: Any) -> str:
    """
    Pull the transcript out of whatever conformer_stream_step returned.

    CTC gives a list of strings; RNNT gives a list of Hypothesis objects. Both
    shapes come back through the same return slot, so both are handled here
    rather than branching on decoder type at every call site.
    """
    if not transcribed:
        return ""
    first = transcribed[0] if isinstance(transcribed, (list, tuple)) else transcribed
    if isinstance(first, str):
        return first.strip()
    return str(getattr(first, "text", "") or "").strip()


def iter_lookaheads() -> Iterator[tuple[str, Lookahead]]:
    """The trained lookaheads, cheapest latency first. For tooling and docs."""
    yield from sorted(LOOKAHEADS.items(), key=lambda kv: kv[1].ms)
