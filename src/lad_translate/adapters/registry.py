"""
Backend resolution.

The pipeline asks for "an STT" and gets one. It never imports a backend, so
moving from the dev Mac to the A4000 is a change of three config strings.

The factory shape is borrowed from VOAG's agent/providers/stt_builder.py and
tts_builder.py, which are the cleanest part of that codebase. What is added
here is the capability report: VOAG has no way to tell whether a given backend
set can meet a latency target, so nobody there can say whether a number is
meaningful. Here every session logs that up front.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..obs.log import get_logger
from .base import BackendCapabilities, MtAdapter, SttAdapter, TtsAdapter

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """What a backend is, and where its timings can be quoted."""

    name: str
    credible_on: frozenset[str]
    """
    Devices where this backend can meet the latency budget.

    Not a single boolean, because credibility is not always a property of the
    model alone. NLLB-200 is far too slow on CPU and entirely comfortable on a
    GPU, so a fixed flag would either bar it from the hardware it is meant for
    or wave it through on hardware it cannot serve.

    Whisper is different: it is a sliding window model whose unstable tail is
    architectural, so no device makes it credible and its set is empty.
    """

    note: str

    def credible(self, device: str) -> bool:
        return device in self.credible_on


# Registered backends. `latency_credible` is a property of the model
# architecture, not of the hardware it runs on: Whisper on an H100 is still a
# sliding window with a long unstable tail.
ANY = frozenset({"cpu", "cuda"})
GPU_ONLY = frozenset({"cuda"})
NEVER = frozenset()

STT_BACKENDS: dict[str, BackendSpec] = {
    "faster-whisper": BackendSpec(
        "faster-whisper",
        credible_on=NEVER,
        note="sliding window, not streaming; about 3.2s at p50 before commit",
    ),
    "fastconformer": BackendSpec(
        "fastconformer",
        credible_on=GPU_ONLY,
        note=(
            "NVIDIA cache-aware streaming transducer, ~114M params. Constant "
            "cost per step at any talk length, which is the property Whisper "
            "cannot have. Lookahead selectable at load: 0/80/480/1040ms, "
            "defaulted here to 480ms rather than NeMo's 1040ms. ENGLISH ONLY "
            "-- 'multi' in the model name means multiple lookaheads, not "
            "multilingual. Adapter written, never run: needs CUDA"
        ),
    ),
    "qwen3-asr": BackendSpec(
        "qwen3-asr",
        credible_on=GPU_ONLY,
        note=(
            "Alibaba, Jan 2026. 0.6B/1.7B, 52 languages, unified streaming and "
            "offline with a 1-8s attention window. Adapter written, never run: "
            "vLLM only, so it cannot be exercised without a GPU"
        ),
    ),
}

MT_BACKENDS: dict[str, BackendSpec] = {
    "opus-mt": BackendSpec(
        "opus-mt", ANY, "CTranslate2 Marian, one model per pair; ~300ms for 4 on CPU"
    ),
    "nllb-200": BackendSpec(
        "nllb-200",
        credible_on=GPU_ONLY,
        note=(
            "200 languages in one model, far better on Indic languages; "
            "~4500ms for 5 on CPU against ~300ms for Opus-MT, so GPU only"
        ),
    ),
    "routing": BackendSpec(
        "routing",
        # Inherits the constraint of whichever backends it actually uses. A set
        # with no Indic languages routes entirely to Opus-MT and is fine on
        # CPU, so the honest static answer is the stricter one and the adapter
        # logs which languages went where.
        credible_on=GPU_ONLY,
        note="per language: Indic to NLLB, the rest to Opus-MT",
    ),
}

TTS_BACKENDS: dict[str, BackendSpec] = {
    "piper": BackendSpec("piper", ANY, "ONNX, streams per sentence, CPU or CUDA"),
    "kokoro": BackendSpec("kokoro", GPU_ONLY, "better quality; no Intel Mac wheel"),
    "chatterbox": BackendSpec(
        "chatterbox",
        credible_on=GPU_ONLY,
        note=(
            "Resemble AI, MIT. 21 languages, one cloned voice across all of "
            "them. NO streaming API in the open-source model, so a whole phrase "
            "renders before anything is heard. No Telugu or Tamil. Adapter "
            "written, never run"
        ),
    ),
}


def build_stt(name: str, **options: object) -> SttAdapter:
    if name == "faster-whisper":
        from .stt_whisper import WhisperSttAdapter

        return WhisperSttAdapter(**options)  # type: ignore[arg-type]
    if name == "fastconformer":
        from .stt_fastconformer import FastConformerSttAdapter

        return FastConformerSttAdapter(**options)  # type: ignore[arg-type]
    if name == "qwen3-asr":
        from .stt_qwen3 import Qwen3SttAdapter

        return Qwen3SttAdapter(**options)  # type: ignore[arg-type]
    raise KeyError(f"unknown STT backend {name!r}; known: {sorted(STT_BACKENDS)}")


def build_mt(name: str, source_language: str, targets: list[str], **options: object) -> MtAdapter:
    if name == "opus-mt":
        from .mt_opus import OpusMtAdapter

        return OpusMtAdapter(source_language, targets, **options)  # type: ignore[arg-type]
    if name == "nllb-200":
        from .mt_nllb import NllbMtAdapter

        return NllbMtAdapter(source_language, targets, **options)  # type: ignore[arg-type]
    if name == "routing":
        from .mt_routing import RoutingMtAdapter

        return RoutingMtAdapter(source_language, targets, **options)  # type: ignore[arg-type]
    raise KeyError(f"unknown MT backend {name!r}; known: {sorted(MT_BACKENDS)}")


def build_tts(name: str, languages: list[str], **options: object) -> TtsAdapter:
    if name == "piper":
        from .tts_piper import PiperTtsAdapter

        return PiperTtsAdapter(languages, **options)  # type: ignore[arg-type]
    if name == "kokoro":
        raise NotImplementedError(
            "kokoro adapter not written yet; its onnxruntime build has no Intel "
            "Mac wheel, so it is a GPU box option"
        )
    if name == "chatterbox":
        from .tts_chatterbox import ChatterboxTtsAdapter

        return ChatterboxTtsAdapter(languages, **options)  # type: ignore[arg-type]
    raise KeyError(f"unknown TTS backend {name!r}; known: {sorted(TTS_BACKENDS)}")


def capabilities(stt: str, mt: str, tts: str, device: str = "cpu") -> BackendCapabilities:
    """
    Report what this backend set can do on this device, for logging at start.

    Credibility is per device. NLLB-200 cannot meet the budget on CPU and can
    on a GPU; Whisper cannot meet it anywhere, because its cost is
    architectural rather than computational.

    A session on a set that is not credible still records its timings. It just
    must never have them quoted as product latency, and this is the flag that
    makes the difference visible rather than assumed.
    """
    specs = (STT_BACKENDS.get(stt), MT_BACKENDS.get(mt), TTS_BACKENDS.get(tts))
    unknown = [n for n, s in zip((stt, mt, tts), specs) if s is None]
    if unknown:
        raise KeyError(f"unknown backend(s): {unknown}")

    credible = all(s.credible(device) for s in specs)  # type: ignore[union-attr]
    notes = [f"{n}: {s.note}" for n, s in zip((stt, mt, tts), specs)]  # type: ignore[union-attr]
    blocking = [
        n for n, s in zip((stt, mt, tts), specs)  # type: ignore[union-attr]
        if not s.credible(device)
    ]
    if blocking:
        notes.append(
            f"LATENCY NOT CREDIBLE on {device}: {', '.join(blocking)} cannot meet "
            "a 2s budget here. Figures are for pipeline validation only."
        )

    caps = BackendCapabilities(
        stt_name=stt,
        mt_name=mt,
        tts_name=tts,
        stt_emits_interims=True,
        latency_credible=credible,
        notes=notes,
    )
    log.info(
        "backend set resolved",
        extra={
            "stt": stt,
            "mt": mt,
            "tts": tts,
            "device": device,
            "latency_credible": credible,
            "blocking": blocking,
            "notes": notes,
        },
    )
    return caps
