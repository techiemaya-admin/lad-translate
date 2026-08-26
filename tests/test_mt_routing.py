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


@pytest.mark.parametrize("language", ["fr", "de", "es", "ar", "zh", "pt"])
def test_everything_else_routes_to_opus(language):
    """Comparable quality at 15x the speed, so it wins where it is good."""
    assert route_for(language) == OPUS


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
