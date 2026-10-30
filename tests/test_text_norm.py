"""
Spoken numbers and spelled letters reach the translator in written form.

Each case is a phrase FastConformer actually produced on the medical test
script, 2026-10-28, and the written form the translator gets right.
"""

from __future__ import annotations

import pytest

from lad_translate.text_norm import spoken_to_written


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("your blood pressure is one hundred and eighty two over one hundred and eight",
         "your blood pressure is 182 over 108"),
        ("febrile at thirty eight point nine degrees celsius", "febrile at 38.9 degrees celsius"),
        ("co amoxiclav one point two grams three times daily", "co amoxiclav 1.2 grams 3 times daily"),
        ("within four and a half hours", "within 4.5 hours"),
        ("in twenty four to forty eight hours", "in 24 to 48 hours"),
        ("bed four is a sixty seven year old female", "bed 4 is a 67 year old female"),
        ("amlodipine five milligrams once daily", "amlodipine 5 milligrams once daily"),
        ("ph seven point one two", "ph 7.12"),
        ("oxygen saturation of eighty nine per cent", "oxygen saturation of 89 per cent"),
        ("type two diabetes", "type 2 diabetes"),
        ("a stroke at sixty and my mother", "a stroke at 60 and my mother"),
    ],
)
def test_quantities_become_digits(spoken, written):
    assert spoken_to_written(spoken) == written


@pytest.mark.parametrize(
    "phrase",
    ["weakness on one side", "around two thirds of patients", "one of the greatest threats",
     "no one", "bicarbonate nine"],
)
def test_small_numbers_that_are_not_quantities_stay_words(phrase):
    assert spoken_to_written(phrase) == phrase


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("arrange an e c g and", "arrange an ECG and"),
        ("with a c t scan of the brain", "with a CT scan of the brain"),
        ("organisms such as m r s a and carbapenem", "organisms such as MRSA and carbapenem"),
        ("a fasting blood glucose h b a one c a lipid profile",
         "a fasting blood glucose HBA1C a lipid profile"),
        ("low h d l cholesterol", "low HDL cholesterol"),
    ],
)
def test_spelled_letters_become_acronyms(spoken, written):
    assert spoken_to_written(spoken) == written


def test_a_lone_article_is_left_alone():
    assert spoken_to_written("a lipid profile and i think") == "a lipid profile and i think"


def test_punctuation_is_kept_around_a_converted_number():
    assert spoken_to_written("It was thirty eight point nine.") == "It was 38.9."


def test_empty_text():
    assert spoken_to_written("") == ""
