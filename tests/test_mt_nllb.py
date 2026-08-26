"""
NLLB adapter.

Slow: the model is 617MB and one CPU call takes seconds. Kept deliberately
thin, and skipped entirely when the model has not been fetched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lad_translate.adapters.mt_nllb import FLORES, NllbMtAdapter

MODEL = Path(__file__).resolve().parent.parent / "models" / "mt" / "nllb-600m"

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.bin").exists(),
    reason="run tools/fetch_mt_models.py --nllb",
)


@pytest.fixture(scope="module")
def nllb():
    adapter = NllbMtAdapter("en", ["fr", "te", "hi"], model_path=MODEL)
    yield adapter
    adapter.close()


def test_flores_codes_name_the_script_not_just_the_language():
    """Urdu is Arabic script, not Latin. Getting this wrong is silent."""
    assert FLORES["ur"] == "urd_Arab"
    assert FLORES["hi"] == "hin_Deva"
    assert FLORES["te"] == "tel_Telu"
    assert FLORES["zh"] == "zho_Hans"


def test_an_unmapped_language_is_refused_at_construction():
    """Better than discovering it mid-event on the first chunk."""
    with pytest.raises(KeyError, match="FLORES"):
        NllbMtAdapter("en", ["xx"], model_path=MODEL)


def test_missing_model_fails_with_a_usable_message():
    with pytest.raises(FileNotFoundError, match="fetch_mt_models"):
        NllbMtAdapter("en", ["fr"], model_path=Path("/nonexistent"))


async def test_telugu_uses_real_words_not_transliteration(nllb):
    """
    The reason to consider NLLB at all. Opus-MT's family model rendered
    "revenue across the sector" as "సెంటర్ అవతల", transliterating "center" and
    dropping "revenue" entirely.
    """
    out = await nllb.translate(
        "Revenue across the sector grew eleven percent last year,", "en", "te"
    )
    assert any("ఀ" <= ch <= "౿" for ch in out), f"not Telugu script: {out!r}"
    assert "ఆదాయం" in out, f"expected the Telugu word for revenue, got: {out!r}"


async def test_no_latin_leaks_into_hindi(nllb):
    """Opus-MT's en-hi emitted 'Ruue' here."""
    out = await nllb.translate(
        "Revenue across the sector grew eleven percent last year,", "en", "hi"
    )
    assert not [c for c in out if c.isascii() and c.isalpha()], f"Latin in output: {out!r}"


async def test_the_fan_out_returns_every_target(nllb):
    """One batched call, every language, which is the point of one model."""
    out = await nllb.translate_many("the results are encouraging.", "en", ["fr", "te", "hi"])
    assert set(out) == {"fr", "te", "hi"}
    assert all(text.strip() for text in out.values())


async def test_unsupported_target_is_dropped_not_faked(nllb):
    out = await nllb.translate_many("hello.", "en", ["fr", "xx"])
    assert "xx" not in out


async def test_empty_input_produces_empty_output(nllb):
    assert await nllb.translate_many("   ", "en", ["fr"]) == {"fr": ""}
