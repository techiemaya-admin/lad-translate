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
    "AudioFrame",
    "BackendCapabilities",
    "Hypothesis",
    "MtAdapter",
    "SpeechChunk",
    "SttAdapter",
    "Translation",
    "TtsAdapter",
    "VoiceSpec",
    "WordTiming",
    "build_mt",
    "build_stt",
    "build_tts",
    "capabilities",
]
