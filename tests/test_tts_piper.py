"""Piper adapter tests. Skipped when voices have not been fetched."""

from __future__ import annotations

from pathlib import Path

import pytest

from lad_translate.adapters.base import VoiceSpec
from lad_translate.adapters.tts_piper import DEFAULT_VOICES, PiperTtsAdapter

VOICE_ROOT = Path(__file__).resolve().parent.parent / "models" / "tts"

pytestmark = pytest.mark.skipif(
    not (VOICE_ROOT / f"{DEFAULT_VOICES['fr']}.onnx").exists(),
    reason="run tools/fetch_tts_voices.py --defaults",
)


@pytest.fixture(scope="module")
def tts():
    return PiperTtsAdapter(["fr"], voice_root=VOICE_ROOT)


def spec(language="fr", speed=1.0) -> VoiceSpec:
    return VoiceSpec(language=language, voice_id=DEFAULT_VOICES[language], speed=speed)


async def collect(tts, text, voice, chunk_id=0):
    return [c async for c in tts.synthesise(text, voice, chunk_id)]


async def test_produces_audio(tts):
    chunks = await collect(tts, "Bonjour a tous et bienvenue.", spec())
    assert chunks
    assert sum(len(c.pcm) for c in chunks) > 0
    assert all(c.sample_rate == tts.sample_rate for c in chunks)


async def test_exactly_one_chunk_is_marked_last(tts):
    """The pipeline closes the track on is_last, so a missing or duplicated
    flag either truncates the phrase or leaves the stream open."""
    chunks = await collect(tts, "Les resultats sont encourageants.", spec())
    assert sum(1 for c in chunks if c.is_last) == 1
    assert chunks[-1].is_last


async def test_chunk_id_and_language_are_carried_through(tts):
    chunks = await collect(tts, "Bonjour.", spec(), chunk_id=42)
    assert all(c.chunk_id == 42 for c in chunks)
    assert all(c.language == "fr" for c in chunks)


async def test_higher_speed_yields_shorter_audio(tts):
    """The drift policy relies on this: speeding up must actually shorten playout."""
    text = "Les revenus dans le secteur ont augmente de onze pour cent l'an dernier."
    normal = sum(c.duration for c in await collect(tts, text, spec(speed=1.0)))
    faster = sum(c.duration for c in await collect(tts, text, spec(speed=1.25)))
    assert faster < normal * 0.95, f"speed had no effect: {normal:.2f}s vs {faster:.2f}s"


async def test_empty_text_produces_nothing(tts):
    assert await collect(tts, "   ", spec()) == []


async def test_unloaded_language_is_rejected(tts):
    assert not tts.supports("de")
    with pytest.raises(KeyError):
        await collect(tts, "Guten Tag.", spec("de"))


def test_missing_voice_file_fails_with_a_usable_message():
    with pytest.raises(FileNotFoundError, match="fetch_tts_voices"):
        PiperTtsAdapter(["fr"], voice_root=Path("/nonexistent"))
