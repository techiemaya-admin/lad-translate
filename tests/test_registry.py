"""Backend resolution and per-device credibility."""

import pytest

from lad_translate.adapters.registry import (
    MT_BACKENDS,
    STT_BACKENDS,
    TTS_BACKENDS,
    capabilities,
)


def test_whisper_is_never_credible_on_any_device():
    """
    Its cost is architectural, not computational: a sliding window model has a
    long unstable tail on a GPU too. A faster device does not fix it.
    """
    for device in ("cpu", "cuda"):
        assert not STT_BACKENDS["faster-whisper"].credible(device)


def test_nllb_is_credible_on_gpu_but_not_cpu():
    """
    Measured: ~4500ms for five languages on CPU against ~300ms for Opus-MT.
    A single boolean would either bar it from the hardware it is built for or
    wave it through on hardware it cannot serve.
    """
    assert not MT_BACKENDS["nllb-200"].credible("cpu")
    assert MT_BACKENDS["nllb-200"].credible("cuda")


def test_opus_mt_is_credible_on_both():
    assert MT_BACKENDS["opus-mt"].credible("cpu")
    assert MT_BACKENDS["opus-mt"].credible("cuda")


def test_the_production_set_is_credible_on_gpu():
    caps = capabilities("fastconformer", "nllb-200", "piper", device="cuda")
    assert caps.latency_credible


@pytest.mark.parametrize(
    "stt,mt,device",
    [
        ("faster-whisper", "opus-mt", "cpu"),
        ("faster-whisper", "nllb-200", "cuda"),
        ("fastconformer", "nllb-200", "cpu"),
    ],
)
def test_an_incredible_set_says_which_backend_blocks_it(stt, mt, device):
    caps = capabilities(stt, mt, "piper", device=device)
    assert not caps.latency_credible
    assert any("NOT CREDIBLE" in note for note in caps.notes)


def test_unknown_backend_is_refused():
    with pytest.raises(KeyError):
        capabilities("nonexistent", "opus-mt", "piper")


def test_every_registered_backend_declares_its_devices():
    for table in (STT_BACKENDS, MT_BACKENDS, TTS_BACKENDS):
        for name, spec in table.items():
            assert spec.credible_on <= {"cpu", "cuda"}, f"{name} has an odd device set"
            assert spec.note, f"{name} has no note"
