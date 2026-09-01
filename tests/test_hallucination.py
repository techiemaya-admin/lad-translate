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
    is_repetition_loop,
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


# --- the model's own confidence floor ---------------------------------------
#
# Measured live with the speaker silent, every kept segment's signals logged.
# no_speech_prob was useless: 0.013 to 0.318 on invented text, so the
# "confidently non-speech" branch never fired. avg_logprob separated them.


@pytest.mark.parametrize(
    ("phrase", "no_speech", "logprob"),
    [
        ("Me too?", 0.013, -1.431),
        ("I'll explain it to you.", 0.029, -1.331),
        ("Matthew.", 0.318, -1.284),
        ("or you'll be sick of it.", 0.025, -1.207),
        ("Alvin Dab.", 0.135, -1.098),
        ("more kebab", 0.088, -1.014),
    ],
)
def test_segments_below_the_confidence_floor_are_dropped(phrase, no_speech, logprob):
    """Recorded from a live session where nobody was speaking."""
    assert is_hallucination(phrase, no_speech, logprob)


@pytest.mark.parametrize(
    ("phrase", "no_speech", "logprob"),
    [
        ("Hello.", 0.156, -0.877),
        ("Bye bye.", 0.008, -0.752),
        ("or you do it.", 0.275, -0.632),
        ("and all good morning.", 0.061, -0.630),
        ("At the main.", 0.181, -0.594),
        ("I'll think that.", 0.224, -0.580),
    ],
)
def test_segments_above_the_floor_still_reach_the_audience(phrase, no_speech, logprob):
    """Recorded from the same session, and these are the speaker."""
    assert not is_hallucination(phrase, no_speech, logprob)


def test_the_floor_matches_what_whisper_is_told():
    """
    A floor the model is given and the adapter ignores is not a floor. If these
    ever diverge, one of them is dead configuration.
    """
    from lad_translate.adapters import stt_whisper

    assert stt_whisper.LOG_PROB_FLOOR == -1.0


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


# --- the four that got past the gate ----------------------------------------
#
# Observed in a live session on a real phone, translated and spoken to the room.
# The list was matched against the raw string, so none of these hit it: two were
# absent, one carried a trailing sentence, one was the phrase looped.


@pytest.mark.parametrize(
    "phrase",
    [
        "I'll see you later.",
        "I'll see you next time.",
        "I hope you enjoyed this video. Thanks.",
        "And I hope you enjoyed this video. I hope you enjoyed this video.",
    ],
)
def test_the_phrases_observed_live_are_dropped_when_doubtful(phrase):
    assert is_hallucination(phrase, no_speech_prob=0.6, avg_logprob=-0.4)


def test_an_ambiguous_farewell_survives_when_confidently_heard():
    """Someone really can say "I'll see you later" into a conference mic."""
    assert not is_hallucination("I'll see you later.", no_speech_prob=0.05, avg_logprob=-0.15)


# --- broadcast sign-offs, dropped on the words alone -------------------------
#
# Reported from a live phone session: "This is the end of the day, and I will
# meet you in the next episode." arrived in the middle of someone reading a
# device manual aloud, and was translated and spoken to the room. It is on no
# list, and Whisper paraphrases this family freely, so the shape is matched
# instead of the string.
#
# No confidence gate. The gate exists because "thank you" is ambiguous; a
# reference to the next episode is not, and these got through precisely because
# the model was confident about them.


@pytest.mark.parametrize(
    "phrase",
    [
        "This is the end of the day, and I will meet you in the next episode.",
        "I'll see you in the next video.",
        "See you in the next episode!",
        "I hope you enjoyed this video.",
        "I hope you enjoyed this video. Thanks.",
        "Don't forget to subscribe.",
        "Please subscribe to my channel.",
        "Thanks for watching!",
        "Subtitles by the Amara.org community",
        "Catch you in the next stream.",
    ],
)
def test_broadcast_sign_offs_are_dropped_however_confident_the_model_is(phrase):
    assert is_hallucination(phrase, no_speech_prob=0.01, avg_logprob=-0.05)


@pytest.mark.parametrize(
    "phrase",
    [
        "Let's watch the video now.",
        "The next episode of this series airs in March.",
        "I'll see you at lunch.",
        "See you in the main hall after the break.",
        "Our channel partners will present next.",
        "This tutorial covers the safety precautions.",
    ],
)
def test_real_speech_about_videos_and_farewells_survives(phrase):
    """
    Both halves are required. A talk that mentions a video, and a farewell that
    mentions no broadcast, are ordinary things to say at a conference.
    """
    assert not is_hallucination(phrase, no_speech_prob=0.05, avg_logprob=-0.15)


# --- normalisation ----------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["Thank you", "Thank you.", "THANK YOU!", "  thank you  ", "Thank  you."],
)
def test_punctuation_and_case_do_not_need_their_own_entries(phrase):
    assert is_hallucination(phrase, no_speech_prob=0.6, avg_logprob=-0.4)


def test_a_leading_filler_does_not_hide_a_stock_phrase():
    assert is_hallucination("And thanks for watching.", 0.6, -0.4)


def test_stripping_filler_cannot_match_a_real_sentence():
    """What remains still has to match in full, so this is not a substring test."""
    assert not is_hallucination("And the quarterly figures are strong.", 0.6, -0.4)


# --- looping ----------------------------------------------------------------


def test_a_looped_phrase_is_a_hallucination_even_if_it_is_not_on_the_list():
    """
    Whisper repeats one phrase for a whole window when it has nothing to work
    with. The shape is the signal, which is what catches the loops nobody has
    seen yet.
    """
    assert is_hallucination(
        "The committee will reconvene. The committee will reconvene.", 0.6, -0.4
    )


def test_a_short_repeated_interjection_is_left_alone():
    """People really do say this, and it is three words of real speech."""
    assert not is_hallucination("No. No. No.", 0.6, -0.4)


# --- decoder loops, dropped on the words alone -------------------------------
#
# Reported live, all four from one session. None is a broadcast sign-off and
# none is on a list. The earlier loop check missed every one of them: it split
# only on ".!?" so comma-separated repeats read as a single sentence, it needed
# every sentence identical, and it sat behind the confidence gate that these
# had already cleared.


@pytest.mark.parametrize(
    "phrase",
    [
        "I'm Cassie, I'm Cassie, I'm Cassie.",
        "I guess you'll. I guess you'll. I guess you'll.",
        "I love you. I love you, I love you. I love you, I love you.",
        "Mehtun Sibya Arkanda. Mehtun Sibya Arkanda. Mehtun Sibya Arkanda. Mehtun.",
    ],
)
def test_decoder_loops_are_dropped_however_confident_the_model_is(phrase):
    assert is_hallucination(phrase, no_speech_prob=0.01, avg_logprob=-0.05)


@pytest.mark.parametrize(
    "phrase",
    [
        # Two of four clauses repeat. Ordinary speech, and it was in the same
        # session as the loops above.
        "Hello, how are you? I'm quick, how are you?",
        "Hello, how are you? I am TG. How are you?",
        # Single words are exempt however often they repeat.
        "No, no, no.",
        "Yes, yes, yes, yes.",
        # A speaker restating a point, not a loop.
        "The deadline is Friday. Please remember, the deadline is Friday.",
        "Safety first. Read the manual. Then begin.",
    ],
)
def test_real_speech_that_repeats_is_kept(phrase):
    assert not is_hallucination(phrase, no_speech_prob=0.05, avg_logprob=-0.15)


def test_a_two_word_clause_still_counts_as_a_loop():
    """"I'm Cassie" is two words, and three of them in a row is not speech."""
    assert is_repetition_loop("I'm Cassie, I'm Cassie, I'm Cassie.")


def test_a_repeated_clause_that_does_not_dominate_is_not_a_loop():
    assert not is_repetition_loop(
        "Good morning. How are you, how are you. We begin now, and welcome everyone."
    )


def test_a_sentence_said_twice_in_real_speech_survives_when_confident():
    assert not is_hallucination(
        "The committee will reconvene. The committee will reconvene.", 0.05, -0.15
    )


def test_empty_text_is_a_hallucination():
    assert is_hallucination("   ", 0.0, 0.0)


@pytest.mark.parametrize("phrase", [".", "...", "!?", "  --  "])
def test_punctuation_only_segments_are_hallucinations(phrase):
    assert is_hallucination(phrase, 0.0, 0.0)


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
