"""
Glossary terms are lifted out of the source and put back after translation.

Opus-MT carries a placeholder through unchanged and in place, so the sentence
around a term is still the model's work. Terms the model got wrong from correct
English on the medical test script: thrombolysis came back as "coagulation"
(تخثر), the opposite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lad_translate.glossary import Glossary

ROOT = Path(__file__).resolve().parent.parent


def _glossary() -> Glossary:
    return Glossary([
        ("ar", "thrombolysis", "إذابة الخثرة"),
        ("ar", "mechanical thrombectomy", "استئصال الخثرة الميكانيكي"),
        ("ar", "thrombectomy", "استئصال الخثرة"),
        ("ar", "alteplase", "ألتيبلاز"),
    ])


def test_terms_become_placeholders_and_come_back():
    g = _glossary()
    found = g.protect("Thrombolysis with alteplase within 4.5 hours", "ar")
    assert found is not None
    assert found.text == "T001 with T002 within 4.5 hours"
    assert Glossary.restore("T001 مع T002 في غضون 4.5 ساعات", found) == (
        "إذابة الخثرة مع ألتيبلاز في غضون 4.5 ساعات"
    )


def test_the_longest_term_wins():
    found = _glossary().protect("mechanical thrombectomy can be offered", "ar")
    assert found is not None
    assert found.slots == ("استئصال الخثرة الميكانيكي",)


def test_whole_words_only():
    assert _glossary().protect("thrombolysises are not a word", "ar") is None


def test_no_terms_means_nothing_to_protect():
    assert _glossary().protect("good morning everyone", "ar") is None
    assert _glossary().protect("thrombolysis", "fr") is None


@pytest.mark.parametrize(
    "translated",
    ["مع ألتيبلاس في غضون",           # a placeholder dropped
     "T001 مع T002 و T003",          # one invented
     "T001 مع T001"],                # one duplicated, one dropped
)
def test_a_damaged_translation_is_refused_so_the_caller_can_retry(translated):
    found = _glossary().protect("thrombolysis with alteplase", "ar")
    assert found is not None
    assert Glossary.restore(translated, found) is None


def test_the_shipped_medical_glossary_loads():
    g = Glossary.load(ROOT / "glossaries" / "medical.tsv")
    assert "ar" in g.languages()
    found = g.protect("atrial fibrillation on apixaban", "ar")
    assert found is not None and len(found.slots) == 2


def test_a_malformed_line_names_its_line_number(tmp_path):
    bad = tmp_path / "g.tsv"
    bad.write_text("# ok\nar\tterm only\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        Glossary.load(bad)
