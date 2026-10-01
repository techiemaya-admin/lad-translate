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


# --- speaking to both Pipers -------------------------------------------------
#
# piper-tts is MIT up to 1.2.0 and GPL-3.0-or-later from 1.5.0. The shipped
# build pins MIT; a Mac developer cannot install it (no arm64 wheel) and runs
# the later one. _stream is the only place the two differ, so it is tested
# against fakes rather than against whichever happens to be installed.


class _OldPiper:
    """MIT 1.2.0: raw bytes straight out of synthesize_stream_raw."""

    def __init__(self):
        self.calls = []

    def synthesize_stream_raw(self, text, length_scale=None, **kw):
        self.calls.append((text, length_scale))
        yield b"\x01\x02"
        yield b"\x03\x04"


class _NewChunk:
    def __init__(self, pcm):
        self.audio_int16_bytes = pcm


class _NewPiper:
    """GPL 1.5.0+: chunk objects carrying .audio_int16_bytes."""

    def __init__(self):
        self.calls = []

    def synthesize(self, text, config):
        self.calls.append((text, getattr(config, "length_scale", None)))
        yield _NewChunk(b"\x01\x02")
        yield _NewChunk(b"\x03\x04")


def test_the_mit_api_yields_raw_pcm():
    from lad_translate.adapters.tts_piper import _stream

    v = _OldPiper()
    assert list(_stream(v, "bonjour", 0.8)) == [b"\x01\x02", b"\x03\x04"]
    assert v.calls == [("bonjour", 0.8)]


def test_both_apis_produce_the_same_bytes():
    """
    The point of the shim: which Piper is installed must not change a single
    sample reaching the room.
    """
    from lad_translate.adapters.tts_piper import _stream

    old = list(_stream(_OldPiper(), "bonjour", 1.0))
    try:
        new = list(_stream(_NewPiper(), "bonjour", 1.0))
    except ImportError:
        pytest.skip("the GPL piper is not installed, so SynthesisConfig is absent")
    assert old == new


def test_the_mit_path_is_preferred_when_both_are_possible():
    """
    synthesize_stream_raw is checked first, so a build carrying both shapes
    uses the MIT call and never imports SynthesisConfig.
    """
    from lad_translate.adapters.tts_piper import _stream

    class Both(_OldPiper):
        def synthesize(self, text, config):  # pragma: no cover - must not run
            raise AssertionError("the GPL path was taken while the MIT one existed")

    assert list(_stream(Both(), "x", 1.0)) == [b"\x01\x02", b"\x03\x04"]


def test_speed_becomes_the_inverse_length_scale():
    """Faster speech is a SHORTER length_scale; inverting it the wrong way
    makes a lagging language lag further."""
    from lad_translate.adapters.tts_piper import _stream

    v = _OldPiper()
    list(_stream(v, "x", 1.0 / 1.25))
    assert v.calls[0][1] == pytest.approx(0.8)
