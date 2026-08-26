"""
Qwen3-ASR and Chatterbox.

Neither can run here: Qwen3-ASR's streaming path is vLLM-only and Chatterbox
needs torch and CUDA. So these test what CAN be tested without the hardware —
the language coverage, the failure modes, and the registry's account of them.
The adapters themselves are unverified until they run on the A4000, and the
suite should not imply otherwise.
"""

from __future__ import annotations

import pytest

from lad_translate.adapters.registry import STT_BACKENDS, TTS_BACKENDS, build_stt, build_tts
from lad_translate.adapters.tts_chatterbox import SUPPORTED, supports_language

# --- language coverage ------------------------------------------------------


@pytest.mark.parametrize("code", ["fr", "de", "es", "ar", "hi", "en"])
def test_chatterbox_covers_the_european_and_hindi_set(code):
    assert supports_language(code)


@pytest.mark.parametrize("code", ["te", "ta", "ml", "kn", "bn", "ur"])
def test_chatterbox_does_not_cover_these_indic_languages(code):
    """
    Piper has voices for Telugu, Tamil and Malayalam. Chatterbox does not, so
    it is a partial replacement rather than a swap, and adopting it means
    carrying per-language TTS routing the way translation already does.
    """
    assert not supports_language(code)


def test_an_unsupported_language_is_refused_at_construction_with_the_remedy():
    """Discovering this when a Telugu listener taps their language is too late."""
    with pytest.raises(KeyError, match="Route those to Piper"):
        build_tts("chatterbox", ["te"])


def test_coverage_set_is_lowercase_iso_codes():
    assert all(c.islower() and 2 <= len(c) <= 3 for c in SUPPORTED)
    assert supports_language("FR"), "lookup should be case-insensitive"


# --- fail fast --------------------------------------------------------------


def test_qwen_reports_a_missing_dependency_at_construction():
    """
    Not when the audience is already in the room. The model loads at session
    start; checking the package exists costs nothing and turns a mid-event
    failure into a configuration error.
    """
    with pytest.raises(RuntimeError, match="vLLM"):
        build_stt("qwen3-asr")


def test_chatterbox_reports_a_missing_dependency_at_construction():
    with pytest.raises(RuntimeError, match="torch"):
        build_tts("chatterbox", ["fr"])


# --- registry honesty -------------------------------------------------------


def test_neither_new_backend_claims_to_work_on_cpu():
    assert STT_BACKENDS["qwen3-asr"].credible_on == frozenset({"cuda"})
    assert TTS_BACKENDS["chatterbox"].credible_on == frozenset({"cuda"})


def test_the_registry_records_that_chatterbox_does_not_stream():
    """
    The widely quoted 200ms figure is Resemble's managed WebSocket service, not
    the open-source model, whose generate() returns a whole waveform. Piper
    yields per sentence and therefore starts speaking sooner. Anyone comparing
    the two on latency needs that written down.
    """
    note = TTS_BACKENDS["chatterbox"].note.lower()
    assert "no streaming" in note


def test_the_registry_records_that_both_are_unrun():
    for spec in (STT_BACKENDS["qwen3-asr"], TTS_BACKENDS["chatterbox"]):
        assert "never run" in spec.note.lower()


def test_whisper_remains_credible_nowhere_even_beside_the_new_options():
    """Its window is architectural. Adding better options does not change that."""
    assert STT_BACKENDS["faster-whisper"].credible_on == frozenset()
