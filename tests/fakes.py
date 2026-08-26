"""Fakes for pipeline tests. No room, no models, no network."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from lad_translate.adapters.base import AudioFrame, Hypothesis, SpeechChunk, VoiceSpec
from lad_translate.config import (
    LanguageTarget,
    SessionConfig,
    SessionLimits,
    TenantContext,
)

SAMPLE_RATE = 22_050


def session_config(targets=("fr", "de"), **over) -> SessionConfig:
    return SessionConfig(
        session_id=over.pop("session_id", "11111111-1111-4111-8111-111111111111"),
        tenant=TenantContext(
            tenant_id="22222222-2222-4222-8222-222222222222",
            database_url="postgresql://x/y",
            schema="tenant_test",
        ),
        room_name="room-1",
        event_name="Test Event",
        source_language="en",
        targets=[LanguageTarget(c, f"voice-{c}") for c in targets],
        limits=over.pop("limits", SessionLimits()),
        **over,
    )


class FakeRoom:
    """Records what was published, and lets tests dictate queue depth."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.published: dict[str, list[bytes]] = {}
        self.languages_published: list[str] = []
        self.depths: dict[str, float] = {}
        self.closed = False
        self.push_delay: dict[str, float] = {}

    async def publish_languages(self, languages: list[str]) -> None:
        self.languages_published = list(languages)
        for language in languages:
            self.published[language] = []

    async def source_frames(self) -> AsyncIterator[AudioFrame]:
        # The pipeline drives STT from this; the fake STT ignores it.
        for i in range(3):
            yield AudioFrame(b"\x00\x00" * 160, 16_000, i * 0.01, time.monotonic())

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        delay = self.push_delay.get(language, 0.0)
        if delay:
            await asyncio.sleep(delay)
        self.published.setdefault(language, []).append(pcm)

    def queue_depth(self, language: str) -> float:
        return self.depths.get(language, 0.0)

    async def close(self) -> None:
        self.closed = True


class FakeStt:
    """Emits a scripted hypothesis stream."""

    name = "fake-stt"

    def __init__(self, texts: list[str], interval: float = 0.01) -> None:
        self._texts = texts
        self._interval = interval

    @property
    def required_sample_rate(self) -> int:
        return 16_000

    @property
    def emits_interims(self) -> bool:
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def transcribe(self, frames) -> AsyncIterator[Hypothesis]:
        # Drain the frame stream so the backlog guard is exercised.
        async for _ in frames:
            break
        wall = time.monotonic()
        for i, text in enumerate(self._texts):
            yield Hypothesis(
                text=text,
                is_final=i == len(self._texts) - 1,
                t_audio_start=0.0,
                t_audio_end=(i + 1) * 0.5,
                t_wall=wall + i * self._interval,
                seq=i,
            )


class FakeMt:
    name = "fake-mt"

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._fail_for = fail_for or set()

    def supports(self, source: str, target: str) -> bool:
        return True

    async def translate(self, text: str, source: str, target: str) -> str:
        return f"[{target}] {text}"

    async def translate_many(self, text: str, source: str, targets: list[str]) -> dict[str, str]:
        self.calls.append(text)
        return {t: "" if t in self._fail_for else f"[{t}] {text}" for t in targets}


class FakeTts:
    """Records the speed it was asked for, and can be made to fail."""

    name = "fake-tts"

    def __init__(self, sample_rate: int = SAMPLE_RATE, fail_for: set[str] | None = None) -> None:
        self._sample_rate = sample_rate
        self.speeds: dict[str, list[float]] = {}
        self.spoken: dict[str, list[str]] = {}
        self._fail_for = fail_for or set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def supports(self, language: str) -> bool:
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def synthesise(
        self, text: str, voice: VoiceSpec, chunk_id: int
    ) -> AsyncIterator[SpeechChunk]:
        self.speeds.setdefault(voice.language, []).append(voice.speed)
        if voice.language in self._fail_for:
            raise RuntimeError(f"synthesis failed for {voice.language}")
        self.spoken.setdefault(voice.language, []).append(text)
        yield SpeechChunk(
            pcm=b"\x00\x00" * 1000,
            sample_rate=self._sample_rate,
            chunk_id=chunk_id,
            language=voice.language,
            is_last=True,
            t_wall=time.monotonic(),
        )
