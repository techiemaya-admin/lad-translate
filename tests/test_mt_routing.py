"""
Per-language translation routing.

The routing table itself is pure and always tested. The end-to-end tests need
both model sets and skip without them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lad_translate.adapters.mt_routing import (
    DEFAULT_ROUTES,
    NLLB,
    OPUS,
    RoutingMtAdapter,
    route_for,
)

MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "mt"
HAVE_NLLB = (MODEL_ROOT / "nllb-600m" / "model.bin").exists()
HAVE_OPUS = (MODEL_ROOT / "en-fr" / "model.bin").exists()


# --- routing table (no models needed) ---------------------------------------


@pytest.mark.parametrize("language", ["hi", "te", "ta", "ml", "kn"])
def test_indic_languages_route_to_nllb(language):
    """
    Opus-MT hallucinated on short input for Telugu and produced a semantic
    inversion for Hindi. Tamil, Malayalam and Kannada share the same en-dra
    family model as Telugu, so the failure belongs to the model.
    """
    assert route_for(language) == NLLB


@pytest.mark.parametrize("language", ["fr", "de", "es", "zh", "pt"])
def test_everything_else_routes_to_opus(language):
    """Comparable quality and cheaper, so it wins where it is good."""
    assert route_for(language) == OPUS


def test_arabic_routes_to_opus_because_nllb_cannot_be_sold():
    """
    Arabic is on Opus-MT for a LICENCE reason, not a quality one.

    It went to NLLB at 9da1c79 because Opus-MT looped: "that the world has
    seen", 23 characters, came back as 212 characters of Arabic - the word for
    "the afterlife" repeated about 25 times - and ten times the text is ten
    times the speech, which filled the playout queue and cost the next phrase.
    NLLB translated it correctly.

    NLLB-200 is licensed CC-BY-NC-4.0. A model that may not be used
    commercially cannot ship in a product that is sold, whatever it scores, so
    Arabic comes back here.

    What makes that survivable is that the loop is now BOUNDED, not fixed: see
    mt_opus.MIN_OUTPUT_TOKENS, which replaced a 32-token floor that defeated
    the ratio on exactly the short phrases this chunker emits, and
    trim_degenerate_repetition, which cuts a repeated run. The same input now
    returns 16 characters instead of 212. It is still the weakest language
    here and still sometimes reaches for religious register; it can no longer
    flood playout.
    """
    assert route_for("ar") == OPUS


def test_an_unlisted_language_defaults_to_opus():
    assert route_for("xx") == OPUS


def test_custom_routes_override_the_defaults():
    assert route_for("fr", {"fr": NLLB}) == NLLB
    assert route_for("te", {}) == OPUS


def test_the_table_only_names_known_backends():
    assert set(DEFAULT_ROUTES.values()) <= {OPUS, NLLB}


# --- end to end -------------------------------------------------------------

pytest_mark = pytest.mark.skipif(
    not (HAVE_NLLB and HAVE_OPUS), reason="needs both opus-mt and nllb models"
)


@pytest.fixture(scope="module")
def routed():
    if not (HAVE_NLLB and HAVE_OPUS):
        pytest.skip("needs both opus-mt and nllb models")
    adapter = RoutingMtAdapter(
        "en",
        ["fr", "hi", "te"],
        opus_options={"model_root": MODEL_ROOT},
        nllb_options={"model_path": MODEL_ROOT / "nllb-600m"},
    )
    yield adapter
    adapter.close()


async def test_each_language_reports_its_backend(routed):
    assert routed.backend_for("fr") == OPUS
    assert routed.backend_for("hi") == NLLB
    assert routed.backend_for("te") == NLLB


async def test_one_call_returns_every_language_across_both_backends(routed):
    out = await routed.translate_many("the results are encouraging.", "en", ["fr", "hi", "te"])
    assert set(out) == {"fr", "hi", "te"}
    assert all(text.strip() for text in out.values())


async def test_short_input_no_longer_hallucinates_in_telugu(routed):
    """
    The regression that forced this design. Opus-MT turned four words into
    fifteen words of Tamil-laced nonsense, and the chunker produces short
    chunks by design.
    """
    out = await routed.translate("the hand of God.", "en", "te")
    assert len(out.split()) <= 6, f"output far longer than the input: {out!r}"
    assert not any("஀" <= ch <= "௿" for ch in out), f"Tamil leaked into Telugu: {out!r}"


async def test_hindi_no_longer_inverts_the_meaning(routed):
    """Opus-MT rendered "revolutionary" as मूलतत्त्ववादी, meaning fundamentalist."""
    out = await routed.translate(
        "And yet the same revolutionary beliefs, for which our forebears fought, "
        "are still at issue",
        "en",
        "hi",
    )
    assert "क्रांतिकारी" in out, f"expected the Hindi word for revolutionary: {out!r}"
    assert "मूलतत्त्ववादी" not in out


async def test_a_failing_backend_does_not_silence_the_other(routed):
    """One dead backend must not take down the languages served by the other."""

    class Broken:
        def supports(self, source, target):
            return True

        async def translate_many(self, text, source, targets):
            raise RuntimeError("backend died")

    saved = routed._backends[NLLB]
    try:
        routed._backends[NLLB] = Broken()
        out = await routed.translate_many("hello there.", "en", ["fr", "hi", "te"])
        assert out["fr"].strip(), "French went silent because NLLB failed"
        assert out["hi"] == "" and out["te"] == ""
    finally:
        routed._backends[NLLB] = saved


async def test_unsupported_target_is_dropped(routed):
    out = await routed.translate_many("hello.", "en", ["fr", "xx"])
    assert "xx" not in out


# --- which directory and which token a language gets --------------------------
#
# Pure, so these run without any model present - unlike tests/test_mt_opus.py,
# whose module-level skipif skips every test in it when en-fr is absent.


def _resolver(model_root):
    """An adapter with only what _resolve reads. Construction proper loads
    models and needs at least one target, neither of which this is about."""
    from lad_translate.adapters.mt_opus import OpusMtAdapter

    adapter = OpusMtAdapter.__new__(OpusMtAdapter)
    adapter.model_root = model_root
    adapter.source_language = "en"
    return adapter


def _pair(root, name):
    (root / name).mkdir()
    (root / name / "model.bin").write_bytes(b"")


def test_arabic_pair_model_is_given_its_target_token(tmp_path):
    """
    eng-ara is one model over several Arabic varieties and its README requires
    a sentence-initial >>id<<. The pair directory exists, so before this the
    resolver returned (en-ar, None) and never read MULTILINGUAL - an "ar" entry
    there would have been a silent no-op.
    """
    _pair(tmp_path, "en-ar")
    directory, token = _resolver(tmp_path)._resolve("ar")
    assert directory == tmp_path / "en-ar"
    assert token == ">>ara<<"


def test_a_family_token_never_attaches_to_a_pair_model(tmp_path):
    """en-ta beats en-dra, and >>tam<< means nothing to a one-pair model."""
    _pair(tmp_path, "en-ta")
    _pair(tmp_path, "en-dra")
    directory, token = _resolver(tmp_path)._resolve("ta")
    assert directory == tmp_path / "en-ta"
    assert token is None


def test_a_family_model_is_used_with_its_token_when_no_pair_exists(tmp_path):
    _pair(tmp_path, "en-dra")
    directory, token = _resolver(tmp_path)._resolve("te")
    assert directory == tmp_path / "en-dra"
    assert token == ">>tel<<"


def test_a_language_with_no_entry_gets_no_token(tmp_path):
    _pair(tmp_path, "en-fr")
    assert _resolver(tmp_path)._resolve("fr") == (tmp_path / "en-fr", None)
