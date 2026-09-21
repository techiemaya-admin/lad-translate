"""
What a thumbs-down may teach, and - more importantly - what it must not.

The dangerous direction is learning too much. A rule derived from an
operator's one-sentence edit fires on every phrase for the rest of the event,
so every case that refuses here is a case where a plausible edit would have
rewritten the speaker's every "a" or "the".
"""

from __future__ import annotations

from lad_translate.feedback import MAX_SPAN_WORDS, derive_rule

# --- the cases that SHOULD learn --------------------------------------------


def test_one_misheard_word_becomes_a_rule():
    d = derive_rule(
        "any emotion I can to love for Irene Adler",
        "any emotion akin to love for Irene Adler",
        "en",
    )
    assert d.learned
    assert d.rule.wrong == "I can"
    assert d.rule.right == "akin"
    assert d.rule.language == "en"
    assert "'I can'" in d.reason and "akin" in d.reason


def test_a_misheard_name_of_two_words_becomes_one_rule():
    d = derive_rule("for I can Adler, all emotions", "for Irene Adler, all emotions", "en")
    assert d.learned
    assert d.rule.wrong == "I can"
    assert d.rule.right == "Irene"


def test_a_target_language_edit_becomes_a_target_rule():
    d = derive_rule("le raisonnement le plus parfait", "le raisonnement le plus précis", "fr")
    assert d.learned
    assert d.rule.language == "fr"
    assert d.rule.wrong == "parfait"
    assert d.rule.right == "précis"


def test_removing_a_hallucinated_word_becomes_a_deletion_rule():
    d = derive_rule("thank you the speaker said", "the speaker said", "en")
    assert d.learned
    assert d.rule.wrong == "thank you"
    assert d.rule.right == ""
    assert "removed" in d.reason


def test_the_wrong_side_keeps_the_produced_casing():
    """The matcher sees the surface form again; keep it as it came."""
    d = derive_rule("went to Dubay yesterday", "went to Dubai yesterday", "en")
    assert d.rule.wrong == "Dubay"
    assert d.rule.right == "Dubai"


def test_a_span_at_the_limit_is_still_learned():
    got = "x " + " ".join(["bad"] * MAX_SPAN_WORDS) + " z"
    want = "x " + " ".join(["good"] * MAX_SPAN_WORDS) + " z"
    assert derive_rule(got, want, "en").learned


# --- the cases that MUST NOT learn ------------------------------------------


def test_a_function_word_is_never_a_rule():
    """'a' -> 'the' is a fine edit and a catastrophic rule."""
    d = derive_rule("he was a machine", "he was the machine", "en")
    assert not d.learned
    assert "too short" in d.reason or "function word" in d.reason


def test_a_longer_function_word_is_still_refused():
    d = derive_rule("that machine was here", "this machine was here", "en")
    assert not d.learned
    assert "function word" in d.reason


def test_a_rephrase_is_kept_as_an_example_not_a_rule():
    got = "the most perfect reasoning and observing machine the world has seen"
    want = "the finest thinking and watching device that anyone has ever known"
    d = derive_rule(got, want, "en")
    assert not d.learned
    assert "rephrase" in d.reason or "separate changes" in d.reason


def test_two_separate_edits_do_not_become_a_rule():
    """A rule fixes one phrase; two edits in one line are two feedbacks."""
    d = derive_rule("Irene Adlur went to Dubay", "Irene Adler went to Dubai", "en")
    assert not d.learned
    assert "2 separate changes" in d.reason


def test_an_insertion_cannot_be_learned():
    d = derive_rule("went to Dubai", "went quickly to Dubai", "en")
    assert not d.learned
    assert "added" in d.reason


def test_casing_only_is_not_a_correction():
    d = derive_rule("irene adler", "Irene Adler", "en")
    assert not d.learned
    assert "casing" in d.reason


def test_empty_expected_teaches_nothing():
    d = derive_rule("something", "   ", "en")
    assert not d.learned
    assert "empty" in d.reason


def test_a_span_past_the_limit_is_refused():
    got = "x " + " ".join(["bad"] * (MAX_SPAN_WORDS + 1)) + " z"
    want = "x " + " ".join(["good"] * (MAX_SPAN_WORDS + 1)) + " z"
    d = derive_rule(got, want, "en")
    assert not d.learned
    assert "rephrase" in d.reason


def test_a_learned_rule_is_a_valid_correction():
    """Whatever comes out here must be constructible by the matcher."""
    from lad_translate.corrections import Corrections

    d = derive_rule("for I can Adler", "for Irene Adler", "en")
    rules = Corrections([d.rule])
    assert rules.apply("I can Adler again", "en") == ("Irene Adler again", 1)
