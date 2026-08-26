"""
Whisper silence hallucinations.

Whisper emits stock phrases over silence and room tone, because its training
data is full of YouTube audio. Observed in a live session: "Thanks for
watching!" and a long run of bare "Thank you." where nobody said either.
Translated and spoken to an audience, an invented politeness is worse than a
gap.

The filter is a pure function so it can be tested against the signals directly,
without having to reproduce the audio that triggers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lad_translate.adapters.stt_whisper import (
    HALLUCINATED_ON_SILENCE,
    is_hallucination,
    resample_to_16k,
)

# --- confidently non-speech -------------------------------------------------


def test_confident_non_speech_is_dropped_whatever_it_said():
    """The model saying "this was not speech" and producing words anyway is
    the exact shape of the failure, regardless of the words."""
    assert is_hallucination("The quarterly figures are strong.", 0.95, -0.9)


def test_low_confidence_alone_is_not_enough():
    """Real speech transcribed badly still deserves to reach the audience."""
    assert not is_hallucination("mumbled something here", 0.2, -0.9)


def test_high_no_speech_alone_is_not_enough():
    assert not is_hallucination("the results are encouraging", 0.9, -0.2)


# --- the blocklist ----------------------------------------------------------


@pytest.mark.parametrize("phrase", ["Thanks for watching!", "Thank you.", "Bye."])
def test_stock_phrases_are_dropped_when_the_segment_is_doubtful(phrase):
    assert is_hallucination(phrase, no_speech_prob=0.6, avg_logprob=-0.4)


@pytest.mark.parametrize("phrase", ["Thank you.", "Bye.", "Okay."])
def test_the_same_phrases_survive_when_confidently_heard(phrase):
    """
    People genuinely say these. A blanket blocklist would delete real speech
    to remove an artefact, so the phrase only counts as evidence when the
    segment already looks like non-speech.
    """
    assert not is_hallucination(phrase, no_speech_prob=0.05, avg_logprob=-0.15)


def test_a_real_sentence_containing_thank_you_is_never_matched():
    """Matching is on the whole segment, not a substring."""
    assert not is_hallucination(
        "Thank you all for coming to the summit today.", 0.55, -0.75
    )


def test_empty_text_is_a_hallucination():
    assert is_hallucination("   ", 0.0, 0.0)


def test_blocklist_entries_are_lowercase_for_matching():
    assert all(p == p.lower() for p in HALLUCINATED_ON_SILENCE)


# --- energy gate ------------------------------------------------------------


def test_resampling_preserves_silence():
    """The energy gate reads the resampled buffer, so silence must stay silent."""
    quiet = np.zeros(16000, dtype=np.int16).tobytes()
    out = resample_to_16k(quiet, 22050)
    assert float(np.sqrt(np.mean(np.square(out)))) == 0.0


def test_resampling_preserves_level_for_real_signal():
    tone = (np.sin(np.linspace(0, 200 * np.pi, 22050)) * 8000).astype(np.int16).tobytes()
    out = resample_to_16k(tone, 22050)
    level = float(np.sqrt(np.mean(np.square(out))))
    assert 0.1 < level < 0.5, f"level changed unexpectedly: {level}"
