"""
Qwen3-ASR speech recognition. GPU BACKEND, NOT YET RUN ON HARDWARE.

Written against the documented streaming API but never executed: the streaming
path requires vLLM, which has no CPU build, so this cannot be exercised on the
development machine at all. Treat every line as unverified until it has run on
the A4000. It is here so the work is waiting when the hardware is.

WHY IT MATTERS

Whisper is a 30 second window model. Every streaming wrapper around it
re-transcribes a sliding buffer, which is why this project carries the
constraint

    emit_interval > window_s * RTF

and why Whisper small shed 74% of the audio on two cores. That cost is
architectural: a faster device does not remove the window.

Qwen3-ASR does streaming and offline inference in one model with a 1 to 8
second attention window, generating incrementally rather than re-reading a
buffer. If that holds up under load it removes the constraint rather than
loosening it.

WHAT IT COSTS

    vLLM only          no CPU, no ONNX, no CTranslate2
    no timestamps      "streaming inference does not support batch inference
                       or returning timestamps"
    0.6B or 1.7B       against FastConformer's ~114M, and it shares the GPU
                       with translation and every TTS voice

The missing timestamps are survivable. Hypothesis.words is optional and the
chunker interpolates across the hypothesis span when it is absent. That costs
precision in the latency figures, which is a real loss on a project whose whole
argument is that measured beats estimated.

A NICE FIT

The model exposes `unfixed_chunk_num` and `unfixed_token_num`: its own notion
of which trailing tokens are still revisable. That is the same idea as the
chunker's LocalAgreement stability window, arrived at from the other end. Worth
measuring whether the model's own instability signal beats the chunker's
external one, rather than running both blind.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator
from typing import Self

import numpy as np

from ..obs.log import get_logger
from .audio import SAMPLE_RATE_16K, resample_to_16k
from .base import AudioFrame, Hypothesis, SttAdapter

log = get_logger(__name__)

QWEN_SAMPLE_RATE = SAMPLE_RATE_16K
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"


class Qwen3SttAdapter(SttAdapter):
    """Streaming ASR that generates incrementally instead of re-reading a buffer."""

    name = "qwen3-asr"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        chunk_size_s: float = 1.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        gpu_memory_utilization: float = 0.4,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_name = model
        self.language = language

        self.chunk_size_s = chunk_size_s
        """
        Audio fed per step. This is the latency knob, and it is NOT the same
        thing as Whisper's window: the model keeps its context in a KV cache
        rather than re-reading the last N seconds, so a smaller chunk costs
        latency without multiplying compute.
        """

        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        """How much of the tail the model treats as still revisable."""

        self.gpu_memory_utilization = gpu_memory_utilization
        """
        Deliberately well below vLLM's 0.9 default. This model shares the card
        with the translation backend and every TTS voice, and vLLM claims its
        fraction on start-up whether it needs it or not.
        """

        self.max_new_tokens = max_new_tokens

        # Fail here, not when the audience is already in the room. Loading the
        # model happens in __aenter__ at session start; checking that the
        # package merely EXISTS costs nothing and moves a missing dependency
        # from a mid-event failure to a configuration error.
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError(
                "qwen_asr is not installed. It needs vLLM and a CUDA device; "
                "there is no CPU build, so this backend cannot run on the "
                "development machine."
            )

        self._model = None
        self._state = None
        self._seq = 0
        self._audio_position = 0.0

    # -------------------------------------------------------------------------

    @property
    def required_sample_rate(self) -> int:
        return QWEN_SAMPLE_RATE

    @property
    def emits_interims(self) -> bool:
        return True

    async def __aenter__(self) -> Self:
        from qwen_asr import Qwen3ASRModel

        started = time.monotonic()
        self._model = await asyncio.to_thread(
            Qwen3ASRModel.LLM,
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_new_tokens=self.max_new_tokens,
        )
        self._state = self._model.init_streaming_state(
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
            chunk_size_sec=self.chunk_size_s,
        )
        log.info(
            "Qwen3-ASR loaded",
            extra={
                "model": self.model_name,
                "chunk_size_s": self.chunk_size_s,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "load_s": round(time.monotonic() - started, 2),
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._model is not None and self._state is not None:
            await asyncio.to_thread(self._model.finish_streaming_transcribe, self._state)
        self._model = None
        self._state = None

    # -------------------------------------------------------------------------

    async def transcribe(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        if self._model is None or self._state is None:
            raise RuntimeError("use Qwen3SttAdapter as an async context manager")

        samples_per_step = int(QWEN_SAMPLE_RATE * self.chunk_size_s)
        buffer = np.zeros(0, dtype=np.float32)
        last_text = ""

        async for frame in frames:
            buffer = np.concatenate([buffer, resample_to_16k(frame.pcm, frame.sample_rate)])
            while buffer.size >= samples_per_step:
                segment, buffer = buffer[:samples_per_step], buffer[samples_per_step:]
                text = await self._feed(segment)
                self._audio_position += self.chunk_size_s
                # state.text is cumulative from session start, which is exactly
                # what the chunker expects. Only emit when it actually moved:
                # a hypothesis identical to the last one tells the stability
                # tracker nothing and costs it a slot in its agreement window.
                if text and text != last_text:
                    last_text = text
                    yield self._hypothesis(text, is_final=False)

        if buffer.size:
            text = await self._feed(buffer)
            self._audio_position += buffer.size / QWEN_SAMPLE_RATE
            if text:
                last_text = text
        if last_text:
            yield self._hypothesis(last_text, is_final=True)

    async def _feed(self, segment: np.ndarray) -> str:
        def run() -> str:
            self._model.streaming_transcribe(segment, self._state)
            return (getattr(self._state, "text", "") or "").strip()

        return await asyncio.to_thread(run)

    def _hypothesis(self, text: str, is_final: bool) -> Hypothesis:
        self._seq += 1
        return Hypothesis(
            text=text,
            is_final=is_final,
            t_audio_start=0.0,
            t_audio_end=self._audio_position,
            t_wall=time.monotonic(),
            seq=self._seq,
            language=getattr(self._state, "language", None) or self.language,
            # No timestamps in streaming mode, so the chunker interpolates.
            words=None,
        )
