"""Backend adapters. The pipeline talks to protocols here, never to a backend."""

from .base import (
    AudioFrame,
    BackendCapabilities,
    Hypothesis,
    MtAdapter,
    SpeechChunk,
    SttAdapter,
    Translation,
    TtsAdapter,
    VoiceSpec,
    WordTiming,
)
from .registry import build_mt, build_stt, build_tts, capabilities

__all__ = [
    "AudioFrame", "Hypothesis", "WordTiming", "SpeechChunk", "VoiceSpec",
    "Translation", "BackendCapabilities",
    "SttAdapter", "MtAdapter", "TtsAdapter",
    "build_stt", "build_mt", "build_tts", "capabilities",
]
