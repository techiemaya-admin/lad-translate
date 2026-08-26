"""
Piper text to speech.

Chosen for the CPU dev machine, but it carries over: PiperVoice.load takes
use_cuda, so the same adapter serves the A4000. Kokoro is the better sounding
GPU option and its onnxruntime build has no Intel Mac wheel, which is why it is
not the local default.

MEASURED on the dev Mac, fr_FR-siwis-medium, once warm:

    words   time to first audio    real time factor
    5             266ms                  0.13
    6             338ms                  0.12
    12            468ms                  0.13

Two things follow from that table.

First, time to first audio scales with chunk length, because Piper synthesises
a whole sentence before releasing any of it. That couples the chunker's
max_words directly to TTS latency: longer chunks translate better and start
speaking later. Sweep both together, not separately.

Second, an RTF of 0.12 means one stream costs about an eighth of a core. Five
concurrent streams need roughly two thirds of a core on this machine, which is
why the dev Mac is scoped to one or two languages rather than five.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..obs.log import get_logger
from .base import SpeechChunk, TtsAdapter, VoiceSpec

log = get_logger(__name__)

DEFAULT_VOICE_ROOT = Path("models/tts")

# Piper voice per language. The audience hears these, so they are a product
# decision, not a default: pick for clarity at conference volume through cheap
# earbuds, not for warmth.
DEFAULT_VOICES: dict[str, str] = {
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "es": "es_ES-davefx-medium",
    "ar": "ar_JO-kareem-medium",
    # Piper ships three Telugu voices: maya, padmavathi, venkatesh.
    # Pick by listening, not by name.
    "te": "te_IN-maya-medium",
    # Piper ships pratham, priyamvada and rohan for Hindi.
    "hi": "hi_IN-pratham-medium",
}


class PiperTtsAdapter(TtsAdapter):
    """Synthesises one voice per target language, streaming chunks as they land."""

    name = "piper"

    def __init__(
        self,
        languages: list[str],
        voice_root: Path | str = DEFAULT_VOICE_ROOT,
        voices: dict[str, str] | None = None,
        use_cuda: bool = False,
    ) -> None:
        self.voice_root = Path(voice_root)
        self.use_cuda = use_cuda
        self._names = {**DEFAULT_VOICES, **(voices or {})}
        self._voices: dict[str, object] = {}
        self._sample_rate: int | None = None
        # One worker per language. Each Piper session is single threaded, so
        # concurrency across languages comes from here.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(languages)), thread_name_prefix="tts"
        )
        self._load(languages)

    # -------------------------------------------------------------------------

    def _load(self, languages: list[str]) -> None:
        from piper import PiperVoice

        for language in languages:
            name = self._names.get(language)
            if name is None:
                raise KeyError(
                    f"no Piper voice configured for {language!r}. Add one to "
                    "DEFAULT_VOICES or pass voices={...}."
                )
            path = self.voice_root / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"voice {name} not found at {path}. Fetch it with: "
                    f"python tools/fetch_tts_voices.py --voice {name}"
                )
            started = time.monotonic()
            voice = PiperVoice.load(str(path), use_cuda=self.use_cuda)
            self._voices[language] = voice

            rate = voice.config.sample_rate
            if self._sample_rate is None:
                self._sample_rate = rate
            elif rate != self._sample_rate:
                # Mixed rates would need per-track resampling before publishing.
                raise ValueError(
                    f"voice {name} runs at {rate}Hz but the session is at "
                    f"{self._sample_rate}Hz; pick voices with one sample rate"
                )
            log.info(
                "voice loaded",
                extra={
                    "language": language,
                    "voice": name,
                    "sample_rate": rate,
                    "cuda": self.use_cuda,
                    "load_s": round(time.monotonic() - started, 2),
                },
            )

    # -------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("no voices loaded")
        return self._sample_rate

    def supports(self, language: str) -> bool:
        return language in self._voices

    async def __aenter__(self) -> PiperTtsAdapter:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._executor.shutdown(wait=False)

    # -------------------------------------------------------------------------

    async def synthesise(
        self, text: str, voice: VoiceSpec, chunk_id: int
    ) -> AsyncIterator[SpeechChunk]:
        """
        Stream audio for one phrase.

        Piper yields per sentence, so a multi-sentence phrase starts speaking
        before the whole thing is rendered. Each piece is forwarded the moment
        it lands rather than being collected: the audience hears the first one,
        and total synthesis time is irrelevant to them.
        """
        if not self.supports(voice.language):
            raise KeyError(f"no voice loaded for {voice.language!r}")
        if not text.strip():
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def produce() -> None:
            from piper import SynthesisConfig

            # length_scale stretches duration, so it is the inverse of speed.
            # The pipeline raises speed when a language chain falls behind,
            # because translated speech runs longer than the source and the lag
            # compounds across a talk.
            config = SynthesisConfig(
                length_scale=1.0 / voice.speed if voice.speed > 0 else 1.0
            )
            try:
                for piece in self._voices[voice.language].synthesize(text, config):
                    loop.call_soon_threadsafe(queue.put_nowait, piece.audio_int16_bytes)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(self._executor, produce)

        pending: bytes | None = await queue.get()
        while pending is not None:
            nxt = await queue.get()
            yield SpeechChunk(
                pcm=pending,
                sample_rate=self.sample_rate,
                chunk_id=chunk_id,
                language=voice.language,
                is_last=nxt is None,
                t_wall=time.monotonic(),
            )
            pending = nxt

        # Surface synthesis errors rather than ending the stream quietly.
        await future
