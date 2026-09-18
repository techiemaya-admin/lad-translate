"""
Operator corrections, and the ways a naive find-and-replace gets them wrong.

Every case here is one an operator will actually produce at a venue: a name
inside a longer word, two rules that point at each other, a phrase that should
beat the single word inside it, Arabic, an apostrophe, a plus sign.
"""

from __future__ import annotations

import pytest

from lad_translate.corrections import Correction, Corrections, normalise


def rules(*triples) -> Corrections:
    return Corrections([Correction(w, r, lang) for w, r, lang in triples])


# --- the basic promise ------------------------------------------------------


def test_a_misheard_name_is_replaced():
    c = rules(("I can Adler", "Irene Adler", "en"))
    out, hits = c.apply("any emotion akin to love for I can Adler, all emotions", "en")
    assert out == "any emotion akin to love for Irene Adler, all emotions"
    assert hits == 1


def test_every_occurrence_in_the_phrase_is_corrected():
    c = rules(("techie maya", "TechieMaya", "en"))
    out, hits = c.apply("techie maya and techie maya again", "en")
    assert out == "TechieMaya and TechieMaya again"
    assert hits == 2


def test_a_rule_for_another_language_does_not_fire():
    c = rules(("chat", "Chat", "fr"))
    assert c.apply("chat", "en") == ("chat", 0)
    assert c.apply("chat", "fr") == ("Chat", 1)


# --- whole words ------------------------------------------------------------


def test_a_short_rule_does_not_eat_a_longer_word():
    """The bug that makes this feature dangerous: 'ad' inside 'Adler'."""
    c = rules(("ad", "AD", "en"))
    out, hits = c.apply("Adler read the advert", "en")
    assert out == "Adler read the advert"
    assert hits == 0


def test_a_word_is_corrected_next_to_punctuation():
    c = rules(("adler", "Adler", "en"))
    assert c.apply("to adler, and adler.", "en") == ("to Adler, and Adler.", 2)


def test_a_rule_containing_punctuation_matches_literally():
    """Operators type words, not patterns. 'C++' is a name, not a quantifier."""
    c = rules(("c plus plus", "C++", "en"))
    out, hits = c.apply("he wrote it in c plus plus", "en")
    assert out == "he wrote it in C++"
    assert hits == 1


def test_a_regex_metacharacter_is_not_a_pattern():
    c = rules(("a.b", "AB", "en"))
    assert c.apply("axb", "en") == ("axb", 0)
    assert c.apply("a.b", "en") == ("AB", 1)


def test_an_apostrophe_survives():
    c = rules(("lad s", "LAD's", "en"))
    assert c.apply("the lad s console", "en") == ("the LAD's console", 1)


# --- precedence and termination ---------------------------------------------


def test_the_longer_phrase_wins_over_a_word_inside_it():
    c = rules(("irene", "Irene", "en"), ("irene adler", "Irene Adler", "en"))
    out, hits = c.apply("for irene adler", "en")
    assert out == "for Irene Adler"
    assert hits == 1


def test_two_rules_pointing_at_each_other_terminate():
    """
    colour -> color and color -> colour is a loop for anything that applies
    rules repeatedly. One pass means each position is decided once.
    """
    c = rules(("colour", "color", "en"), ("color", "colour", "en"))
    out, hits = c.apply("the colour of the color", "en")
    assert out == "the color of the colour"
    assert hits == 2


def test_a_replacement_is_not_itself_corrected():
    c = rules(("a", "b", "en"), ("b", "c", "en"))
    assert c.apply("a", "en") == ("b", 0 + 1)


def test_the_first_rule_wins_when_two_target_the_same_words():
    c = rules(("maya", "Maya", "en"), ("maya", "MAYA", "en"))
    assert c.apply("maya", "en") == ("Maya", 1)


# --- casing -----------------------------------------------------------------


def test_matching_ignores_case_and_the_replacement_is_verbatim():
    """
    A name is said at the start of a sentence and in the middle of one. The
    operator writes the casing once and gets it in both places.
    """
    c = rules(("dubai demo", "Dubai Demo", "en"))
    out, hits = c.apply("Dubai Demo, dubai demo, DUBAI DEMO", "en")
    assert out == "Dubai Demo, Dubai Demo, Dubai Demo"
    assert hits == 3


# --- other scripts ----------------------------------------------------------


def test_arabic_is_corrected_and_bounded():
    c = rules(("أدلر", "أدلير", "ar"))
    out, hits = c.apply("حب أدلر كان", "ar")
    assert out == "حب أدلير كان"
    assert hits == 1


def test_arabic_does_not_match_inside_a_longer_word():
    c = rules(("حب", "الحب", "ar"))
    assert c.apply("حبيب", "ar") == ("حبيب", 0)


# --- degenerate input -------------------------------------------------------


def test_a_blank_rule_is_refused_at_construction():
    with pytest.raises(ValueError):
        Correction("   ", "something", "en")
    with pytest.raises(ValueError):
        Correction("word", "x", "  ")


def test_an_empty_set_changes_nothing():
    c = Corrections([])
    assert not c
    assert c.apply("untouched", "en") == ("untouched", 0)


def test_empty_text_is_safe():
    c = rules(("a", "b", "en"))
    assert c.apply("", "en") == ("", 0)


def test_whitespace_in_a_rule_is_normalised():
    """A phrase pasted from a document carries whatever spacing it had."""
    assert normalise("  Irene   Adler ") == "Irene Adler"
    c = rules(("  irene   adler ", "Irene Adler", "en"))
    assert c.apply("for irene adler", "en") == ("for Irene Adler", 1)


def test_the_rule_count_is_bounded():
    from lad_translate.corrections import MAX_RULES

    many = [Correction(f"word{i}", f"W{i}", "en") for i in range(MAX_RULES + 50)]
    assert len(Corrections(many)) == MAX_RULES
