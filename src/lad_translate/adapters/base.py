"""
Adapter contracts for the translation pipeline.

Everything the pipeline touches goes through these protocols. Swapping a CPU
backend for a GPU one (faster-whisper -> Parakeet, Piper -> Kokoro) means
writing a new adapter and changing one config value. The pipeline never
imports a backend directly.

Two clocks are carried on every event, and both are required:

    t_audio  seconds since the start of the source audio stream. Stable across
             restarts and independent of processing speed. Use it to line up a
             transcript with the audio that produced it.

    t_wall   time.monotonic() at the moment the event was produced. Use it, and
             only it, to measure latency.

Glass to glass for one phrase is:

    t_wall(first translated audio frame published)
      - t_wall(source audio for the end of that phrase arrived)

Both ends of that subtraction are recorded by the pipeline, so the number is
measured rather than estimated. See session/latency.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# =============================================================================
# AUDIO
# =============================================================================


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A block of mono PCM audio taken from the source track."""

    pcm: bytes
    """Signed 16-bit little-endian mono samples."""

    sample_rate: int

    t_audio: float
    """Seconds since session start, at the FIRST sample in this frame."""

    t_wall: float
    """time.monotonic() when this frame was received from the room."""

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


# =============================================================================
# SPEECH TO TEXT
# =============================================================================


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Audio position of a single word, when the backend can supply it."""

    word: str
    t_audio_start: float
    t_audio_end: float


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """
    One transcript hypothesis from the STT backend.

    Interim hypotheses get revised. A backend may emit "we are going to" and
    then replace it with "we are going to Dubai" and then with "we're going to
    Dubai next". The chunker's whole job is deciding when a prefix has stopped
    moving. `seq` increases monotonically within a session so revisions can be
    ordered; it does NOT mean the text is additive.
    """

    text: str
    is_final: bool
    t_audio_start: float
    t_audio_end: float
    t_wall: float
    seq: int
    language: str | None = None
    confidence: float | None = None
    words: tuple[WordTiming, ...] | None = None
    """
    Per-word audio positions, if the backend produces them.

    The chunker needs the audio time of a chunk's LAST word to measure
    latency honestly. Without this it interpolates across the hypothesis
    span, which is good enough for tuning but adds error to the p95.
    """


@runtime_checkable
class SttAdapter(Protocol):
    """
    Streaming speech to text.

    Implementations must be usable as an async context manager so model
    handles and worker threads get released when a session ends.
    """

    name: str

    @property
    def required_sample_rate(self) -> int:
        """Sample rate the backend needs. The pipeline resamples to match."""
        ...

    @property
    def emits_interims(self) -> bool:
        """
        False means the backend only produces final transcripts.

        The chunker still works, but it can only commit on finals, so its
        latency floor becomes the backend's endpointing delay. Log this at
        session start: it is the single biggest driver of the latency budget.
        """
        ...

    async def __aenter__(self) -> SttAdapter: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[Hypothesis]:
        """Consume audio frames, yield hypotheses as they become available."""
        ...


# =============================================================================
# MACHINE TRANSLATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class Translation:
    chunk_id: int
    language: str
    text: str
    t_wall: float


@runtime_checkable
class MtAdapter(Protocol):
    """
    Text to text translation.

    `translate_many` exists because the fan-out shape is one source chunk into
    N languages. A backend that can batch those into one call should; one that
    cannot should still implement it so the pipeline has a single code path.
    """

    name: str

    def supports(self, source: str, target: str) -> bool:
        """Whether this backend can serve the pair. Checked at session start."""
        ...

    async def translate(self, text: str, source: str, target: str) -> str: ...

    async def translate_many(
        self, text: str, source: str, targets: list[str]
    ) -> dict[str, str]:
        """Translate one chunk into several languages. Keys are target codes."""
        ...


# =============================================================================
# TEXT TO SPEECH
# =============================================================================


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """A block of synthesised audio for one language."""

    pcm: bytes
    sample_rate: int
    chunk_id: int
    language: str
    is_last: bool
    t_wall: float

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


@dataclass(frozen=True, slots=True)
class VoiceSpec:
    """Which voice to use for a language. Backend-specific id, kept opaque."""

    language: str
    voice_id: str
    speed: float = 1.0
    """
    Playout rate. The pipeline raises this when a language chain falls behind,
    because translated speech is routinely longer than the source and the lag
    compounds across a talk. See session/pipeline.py for the drift policy.
    """


@runtime_checkable
class TtsAdapter(Protocol):
    """
    Streaming text to speech.

    `synthesise` must yield its first chunk as early as the backend allows.
    Time to first audio chunk is what the audience feels; total synthesis time
    is not.
    """

    name: str

    @property
    def sample_rate(self) -> int: ...

    def supports(self, language: str) -> bool: ...

    async def __aenter__(self) -> TtsAdapter: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def synthesise(
        self, text: str, voice: VoiceSpec, chunk_id: int
    ) -> AsyncIterator[SpeechChunk]: ...


# =============================================================================
# CAPABILITY REPORT
# =============================================================================


@dataclass(slots=True)
class BackendCapabilities:
    """
    What the resolved backend set can actually do, logged at session start.

    This exists so a session on the dev Mac is never mistaken for a session on
    real hardware. `latency_credible` is False for any CPU backend set: the
    numbers still get recorded, they just must not be quoted as product
    latency.
    """

    stt_name: str
    mt_name: str
    tts_name: str
    stt_emits_interims: bool
    latency_credible: bool
    notes: list[str] = field(default_factory=list)
