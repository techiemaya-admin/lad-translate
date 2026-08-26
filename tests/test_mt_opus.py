"""
Opus-MT adapter tests.

These use the real models under models/mt, so they are skipped when the models
have not been fetched. Run tools/fetch_mt_models.py first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lad_translate.adapters.mt_opus import OpusMtAdapter

MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "mt"


def available(pair: str) -> bool:
    return (MODEL_ROOT / pair / "model.bin").exists()


pytestmark = pytest.mark.skipif(
    not available("en-fr"), reason="run tools/fetch_mt_models.py --pair en-fr"
)


@pytest.fixture(scope="module")
def mt():
    targets = [t for t in ("fr", "de", "es") if available(f"en-{t}")]
    adapter = OpusMtAdapter("en", targets, model_root=MODEL_ROOT)
    yield adapter
    adapter.close()


async def test_translates_a_phrase(mt):
    out = await mt.translate("today we will look at the results.", "en", "fr")
    assert "résultats" in out.lower()


async def test_output_is_not_degenerate(mt):
    """
    Guards the Marian EOS bug.

    Without the end token appended to the source, the decoder never stops and
    emits the same phrase over and over. It is silent and looks like a model
    quality problem. This catches a regression in _Pair.translate.
    """
    out = await mt.translate("Good morning everyone and welcome,", "en", "fr")
    words = out.lower().split()
    assert len(words) < 15, f"suspiciously long output for a short phrase: {out!r}"
    assert words.count("bienvenue,") <= 1, f"repetition loop returned: {out!r}"


async def test_fan_out_returns_every_requested_language(mt):
    targets = list(mt._pairs)
    out = await mt.translate_many("the results are encouraging.", "en", targets)
    assert set(out) == set(targets)
    assert all(text.strip() for text in out.values())


async def test_fan_out_runs_concurrently(mt):
    """
    Fan-out must not be N sequential translations.

    CTranslate2 releases the GIL, so N languages should cost well under N times
    a single translation. Threshold is loose on purpose: this catches the
    executor being removed, not a performance regression.
    """
    targets = list(mt._pairs)
    if len(targets) < 2:
        pytest.skip("needs at least two language pairs")

    phrase = "Revenue across the sector grew eleven percent last year,"
    single = await _time(mt.translate(phrase, "en", targets[0]))
    fan = await _time(mt.translate_many(phrase, "en", targets))
    assert fan < single * len(targets), (
        f"fan-out {fan:.3f}s is not faster than {len(targets)} sequential "
        f"translations at {single:.3f}s each"
    )


async def _time(coro) -> float:
    loop = asyncio.get_running_loop()
    start = loop.time()
    await coro
    return loop.time() - start


async def test_unsupported_pair_is_rejected_not_silently_wrong(mt):
    assert not mt.supports("en", "xx")
    with pytest.raises(KeyError):
        await mt.translate("hello", "en", "xx")


async def test_failed_language_does_not_take_down_the_others(mt):
    """One dead chain must not silence the whole room."""
    targets = [*mt._pairs, "xx"]
    out = await mt.translate_many("the results are encouraging.", "en", targets)
    assert "xx" not in out, "unsupported target is dropped, not faked"
    assert all(out[t].strip() for t in mt._pairs)


def test_missing_model_fails_with_a_usable_message():
    with pytest.raises(FileNotFoundError, match="fetch_mt_models"):
        OpusMtAdapter("en", ["nonexistent"], model_root=MODEL_ROOT)


# --- family models and target tokens ----------------------------------------

TELUGU = "ఀ-౿"
TAMIL = "஀-௿"


def has_script(text: str, lo: str, hi: str) -> bool:
    return any(lo <= ch <= hi for ch in text)


@pytest.fixture(scope="module")
def dravidian():
    if not available("en-dra"):
        pytest.skip("run tools/fetch_mt_models.py --pair en-dra")
    adapter = OpusMtAdapter("en", ["te", "ta"], model_root=MODEL_ROOT)
    yield adapter
    adapter.close()


async def test_telugu_comes_out_in_telugu_script(dravidian):
    out = await dravidian.translate("Good morning everyone and welcome,", "en", "te")
    assert has_script(out, "ఀ", "౿"), f"not Telugu script: {out!r}"


async def test_the_same_model_serves_tamil_via_a_different_token(dravidian):
    out = await dravidian.translate("Good morning everyone and welcome,", "en", "ta")
    assert has_script(out, "஀", "௿"), f"not Tamil script: {out!r}"


async def test_dropping_the_target_token_produces_mixed_languages(dravidian):
    """
    The reason MULTILINGUAL exists.

    en-dra covers four Dravidian languages and picks one from a sentence
    initial token. Without it the model does not fail, it produces a blend:
    Telugu, Kannada and Tamil in a single sentence. Nothing errors, so this
    would reach an audience as confident nonsense.
    """
    pair = dravidian._pairs["te"]
    saved = pair.target_token
    try:
        pair.target_token = None
        blended = await dravidian.translate("Good morning everyone and welcome,", "en", "te")
    finally:
        pair.target_token = saved

    with_token = await dravidian.translate("Good morning everyone and welcome,", "en", "te")
    assert blended != with_token, "the target token had no effect"


async def test_family_languages_share_one_loaded_model(dravidian):
    """
    Telugu and Tamil are the same weights. Loading one copy per language would
    hold four copies of en-dra, which matters at five languages on a 16GB card.
    """
    assert dravidian._pairs["te"].model is dravidian._pairs["ta"].model


async def test_a_dedicated_pair_model_wins_over_the_family_model():
    """en-ta would beat en-dra for Tamil: one pair beats capacity split four ways."""
    if not available("en-fr"):
        pytest.skip("needs en-fr")
    adapter = OpusMtAdapter("en", ["fr"], model_root=MODEL_ROOT)
    try:
        assert adapter._pairs["fr"].target_token is None
        assert adapter._pairs["fr"].model.path.name == "en-fr"
    finally:
        adapter.close()


# --- leading case normalisation ---------------------------------------------


@pytest.fixture(scope="module")
def hindi():
    if not available("en-hi"):
        pytest.skip("run tools/fetch_mt_models.py --pair en-hi")
    adapter = OpusMtAdapter("en", ["hi"], model_root=MODEL_ROOT)
    yield adapter
    adapter.close()


def test_a_capitalised_out_of_vocabulary_word_is_lowercased(hindi):
    """
    spm("Revenue") splits into junk subwords; spm("revenue") is one token.
    Uncorrected the model emits Latin garbage, and Whisper capitalises the
    first word of every chunk.
    """
    pair = hindi._pairs["hi"]
    assert pair._fix_leading_case("Revenue grew.").startswith("revenue")


async def test_the_latin_leak_is_gone(hindi):
    out = await hindi.translate(
        "Revenue across the sector grew eleven percent last year,", "en", "hi"
    )
    latin = [c for c in out if c.isascii() and c.isalpha()]
    assert not latin, f"untranslated Latin characters in output: {out!r}"


def test_acronyms_are_left_alone(hindi):
    pair = hindi._pairs["hi"]
    assert pair._fix_leading_case("NASA confirmed the launch.").startswith("NASA")


def test_a_word_already_in_vocabulary_is_left_alone(hindi):
    """Only fix what is demonstrably broken; do not lowercase on principle."""
    pair = hindi._pairs["hi"]
    text = "Good morning everyone and welcome,"
    assert pair._fix_leading_case(text) == text


def test_an_out_of_vocabulary_proper_noun_is_left_alone(hindi):
    """
    Lowercasing a name loses information. The rule only fires when the
    lowercase form is a known single token, which a rare proper noun is not.
    """
    pair = hindi._pairs["hi"]
    text = "Sharjah is the venue."
    assert pair._fix_leading_case(text) == text


def test_empty_and_single_word_input_is_safe(hindi):
    pair = hindi._pairs["hi"]
    assert pair._fix_leading_case("") == ""
    assert pair._fix_leading_case("   ") == "   "
