"""
Chatterbox speech synthesis. GPU BACKEND, NOT YET RUN ON HARDWARE.

Written against the documented API but never executed. Treat it as unverified
until it has run on the A4000.

A CORRECTION WORTH CARRYING

Chatterbox is often quoted at "200ms to first sample". That figure belongs to
Resemble's MANAGED WebSocket service, not to the open-source model. The
open-source API is

    wav = model.generate(text, language_id="fr")

which returns one complete waveform. There is no streaming method. So time to
first audio equals full synthesis time for the phrase, and it is a larger model
than Piper.

That matters here more than it would elsewhere. Piper already streams per
sentence, and time to first audio was measured at 266ms for a five word phrase
against 468ms for twelve: what the audience feels is when speech STARTS, not
how long the whole phrase takes to render. Chatterbox has to beat Piper on
total synthesis time to break even on latency, not merely on throughput.

WHAT IT BUYS

Better voices, and one cloned voice carried across every language it supports.
On a five language event that is the difference between one interpreter and
five unrelated synthetic strangers, which is a real product quality the current
stack cannot offer at all.

WHAT IT DOES NOT COVER

21 languages, and Telugu is not among them. Neither is Tamil. Piper has all
three. So this is a partial replacement needing the same per language routing
the translation stage already carries, for the same reason: a good model that
does not cover your languages is a second backend, not a swap.

It also embeds a PerTh watermark in its output. Inaudible, and a deliberate
decision someone should take rather than discover.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ..obs.log import get_logger
from .base import SpeechChunk, TtsAdapter, VoiceSpec

log = get_logger(__name__)

# The 21 languages the multilingual model covers. Anything outside this set has
# to fall back to Piper, so the gap is data rather than a comment.
SUPPORTED = frozenset(
    {
        "ar", "cs", "da", "de", "en", "es", "fi", "fr", "he", "hi", "it",
        "ja", "ko", "nl", "no", "pl", "pt", "ru", "sv", "tr", "vi",
    }
)


def supports_language(code: str) -> bool:
    return code.lower() in SUPPORTED


class ChatterboxTtsAdapter(TtsAdapter):
    """Multilingual synthesis with one cloned voice across every language."""

    name = "chatterbox"

    def __init__(
        self,
        languages: list[str],
        reference_voice: Path | str | None = None,
        device: str = "cuda",
        turbo: bool = True,
    ) -> None:
        unsupported = [code for code in languages if not supports_language(code)]
        if unsupported:
            raise KeyError(
                f"Chatterbox does not cover {unsupported}. Route those to Piper: "
                f"it has voices for Telugu, Tamil and Malayalam, which this "
                f"model does not."
            )

        self.languages = list(languages)
        self.device = device
        self.turbo = turbo

        self.reference_voice = Path(reference_voice) if reference_voice else None
        """
        A single clip, 10 seconds or more, cloned across every language.

        Whose voice this is is a decision, not a setting. It will read a
        speaker's words to an audience for the length of an event, so it should
        be chosen and cleared rather than defaulted.
        """

        if self.reference_voice and not self.reference_voice.exists():
            raise FileNotFoundError(f"reference voice not found: {self.reference_voice}")

        # Same reasoning as the Qwen adapter: a missing package should be a
        # configuration error, not a surprise once a session is live.
        if importlib.util.find_spec("chatterbox") is None:
            raise RuntimeError(
                "chatterbox is not installed. It needs torch and a CUDA device."
            )

        self._model = None
        self._sample_rate: int | None = None

    # -------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("model not loaded")
        return self._sample_rate

    def supports(self, language: str) -> bool:
        return language in self.languages

    async def __aenter__(self) -> ChatterboxTtsAdapter:
        started = time.monotonic()

        def load():
            if self.turbo:
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                return ChatterboxTurboTTS.from_pretrained(device=self.device)
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            return ChatterboxMultilingualTTS.from_pretrained(
                device=self.device, t3_model="v3"
            )

        self._model = await asyncio.to_thread(load)
        self._sample_rate = int(self._model.sr)
        log.info(
            "Chatterbox loaded",
            extra={
                "device": self.device,
                "turbo": self.turbo,
                "sample_rate": self._sample_rate,
                "languages": self.languages,
                "cloned_voice": str(self.reference_voice) if self.reference_voice else None,
                "load_s": round(time.monotonic() - started, 2),
            },
        )
        if self.device != "cuda":
            log.warning("Chatterbox on CPU will not meet the latency budget")
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._model = None

    # -------------------------------------------------------------------------

    async def synthesise(
        self, text: str, voice: VoiceSpec, chunk_id: int
    ) -> AsyncIterator[SpeechChunk]:
        """
        Synthesise one phrase.

        Yields exactly ONE chunk, because the model has no streaming method:
        generate() returns the whole waveform. Piper yields per sentence and so
        starts speaking sooner on a multi-sentence phrase. If this backend is
        adopted, the chunker's max_words matters more, not less, because chunk
        length now sets time to first audio directly.

        VoiceSpec.speed is not applied: the API exposes no rate control, so the
        drift controller loses its cheap lever here and can only skip. That is
        a real regression for a language that expands, and Arabic already
        peaked within 20ms of the skip threshold on Piper.
        """
        if self._model is None:
            raise RuntimeError("use ChatterboxTtsAdapter as an async context manager")
        if not self.supports(voice.language):
            raise KeyError(f"no Chatterbox language loaded for {voice.language!r}")
        if not text.strip():
            return

        if voice.speed != 1.0:
            log.debug(
                "Chatterbox cannot vary playout rate; drift correction is limited "
                "to skipping for this language",
                extra={"language": voice.language, "requested_speed": voice.speed},
            )

        def render() -> bytes:
            kwargs: dict = {"language_id": voice.language}
            if self.reference_voice:
                kwargs["audio_prompt_path"] = str(self.reference_voice)
            wav = self._model.generate(text, **kwargs)
            # torch tensor to int16 PCM, without importing torch here: the
            # tensor exposes the array interface numpy needs.
            samples = np.asarray(wav).astype(np.float32).reshape(-1)
            clipped = np.clip(samples, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16).tobytes()

        pcm = await asyncio.to_thread(render)
        if not pcm:
            return
        yield SpeechChunk(
            pcm=pcm,
            sample_rate=self.sample_rate,
            chunk_id=chunk_id,
            language=voice.language,
            is_last=True,
            t_wall=time.monotonic(),
        )
