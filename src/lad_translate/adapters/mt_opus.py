"""
Opus-MT translation through CTranslate2.

Dedicated neural machine translation, not a general LLM. For a two second
budget with five parallel language chains this is the right shape: a small
per-pair model, roughly 75MB, translating a phrase in tens of milliseconds.
A 7B LLM doing the same work costs several hundred milliseconds and a GPU it
would otherwise not need.

The trade is real. NMT handles pronouns, domain terminology and long-range
context worse than an LLM. The answer is a glossary, not a bigger model in the
hot path.

MEASURED on the dev Mac (2 core Haswell, int8): 41ms per phrase, greedy.
Model load is about 1.5s per pair, so pairs are opened at session start and
held, never opened per chunk.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..obs.log import get_logger
from .base import MtAdapter

log = get_logger(__name__)

DEFAULT_MODEL_ROOT = Path("models/mt")

MULTILINGUAL: dict[str, tuple[str, str]] = {
    # Some Opus-MT models cover a language family rather than a pair, and pick
    # the target with a sentence-initial token. Telugu has no dedicated en-te
    # model, only en-dra covering the Dravidian family.
    #
    # Omit the token and the model still translates, into whichever family
    # member it feels like. That is a silent wrong-language bug, not an error.
    "te": ("en-dra", ">>tel<<"),
    "ta": ("en-dra", ">>tam<<"),
    "ml": ("en-dra", ">>mal<<"),
    "kn": ("en-dra", ">>kan<<"),
}
"""target language code -> (model directory, sentence-initial token)"""


class _Model:
    """
    A loaded CTranslate2 model, shared by every target it serves.

    A family model like en-dra serves four languages. Loading it once per
    language would hold four copies of the same weights, which matters when the
    fan-out is five languages on a 16GB card.
    """

    __slots__ = ("eos", "path", "sp_source", "sp_target", "translator")

    def __init__(self, path: Path, device: str, compute_type: str, sharers: int):
        import ctranslate2
        import sentencepiece as spm

        self.path = path
        self.translator = ctranslate2.Translator(
            str(path),
            device=device,
            compute_type=compute_type,
            # One worker per sharer so concurrent calls into a shared model do
            # not serialise behind each other.
            inter_threads=max(1, sharers),
            intra_threads=2,
        )
        self.sp_source = spm.SentencePieceProcessor(str(path / "source.spm"))
        self.sp_target = spm.SentencePieceProcessor(str(path / "target.spm"))
        config = json.loads((path / "config.json").read_text())
        self.eos = config.get("eos_token", "</s>")


class _Pair:
    """One target language, and the model that serves it."""

    __slots__ = ("model", "source", "target", "target_token")

    def __init__(self, source: str, target: str, model: _Model, target_token: str | None):
        self.source = source
        self.target = target
        self.model = model
        self.target_token = target_token
    def _fix_leading_case(self, text: str) -> str:
        """
        Lowercase the first word when capitalising it breaks tokenisation.

        Some Opus-MT vocabularies do not carry the capitalised form of common
        words. In en-hi:

            spm("Revenue") -> ["_R", "even", "ue"]     out of vocabulary
            spm("revenue") -> ["_revenue"]             one clean token

        The model cannot translate the first and emits Latin garbage: "Ruue".
        Lowercased it produces the correct word. Whisper capitalises every
        sentence, so this hits the FIRST WORD OF EVERY CHUNK.

        The rule is deliberately narrow: apply only when the capitalised form
        splits into several pieces AND the lowercase form is a single known
        token. That is strong evidence the lowercase is a real vocabulary entry
        and the capital is not. A proper noun that is out of vocabulary either
        way fails both halves of the test and is left alone, as are acronyms.
        """
        parts = text.split(maxsplit=1)
        if not parts:
            return text
        first = parts[0]
        if first.isupper():
            return text  # acronym: NASA, UAE
        lowered = first[:1].lower() + first[1:]
        if lowered == first:
            return text

        sp = self.model.sp_source
        if len(sp.encode(first, out_type=str)) < 2:
            return text  # capitalised form is already a clean token
        if len(sp.encode(lowered, out_type=str)) != 1:
            return text  # lowercase is not a known word either

        return lowered if len(parts) == 1 else f"{lowered} {parts[1]}"

    def translate(self, text: str, beam_size: int, max_output_ratio: float) -> str:
        model = self.model
        tokens = model.sp_source.encode(self._fix_leading_case(text), out_type=str)
        if self.target_token:
            # Must lead the sequence, and must NOT go through SentencePiece:
            # the tokeniser would split ">>tel<<" into pieces the model does
            # not recognise as a language selector.
            tokens = [self.target_token, *tokens]
        # Marian models require the source sequence to be terminated. Without
        # it the decoder never stops and emits degenerate repetition:
        # "Bonjour Bonjour Bonjour, bienvenue, bienvenue, bienvenue..."
        # It is silent, it looks like a quality problem, and it is not.
        tokens = [*tokens, model.eos]
        # Bound the output relative to the input. Belt and braces against
        # runaway decoding on a malformed model.
        max_len = max(32, int(len(tokens) * max_output_ratio))
        result = model.translator.translate_batch(
            [tokens], beam_size=beam_size, max_decoding_length=max_len
        )
        return model.sp_target.decode(result[0].hypotheses[0])


class OpusMtAdapter(MtAdapter):
    """Translates one source language into several targets, concurrently."""

    name = "opus-mt"

    def __init__(
        self,
        source_language: str,
        targets: list[str],
        model_root: Path | str = DEFAULT_MODEL_ROOT,
        device: str = "cpu",
        compute_type: str | None = None,
        beam_size: int = 1,
        max_output_ratio: float = 3.0,
    ) -> None:
        self.source_language = source_language
        self.model_root = Path(model_root)
        self.device = device
        # int8 on CPU measured 4x faster than float32 with no quality loss on
        # these models. float16 is the equivalent choice on CUDA.
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.beam_size = beam_size
        """1 is greedy. Raising it improves phrasing and costs roughly 3x."""

        self.max_output_ratio = max_output_ratio
        self._pairs: dict[str, _Pair] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=len(targets), thread_name_prefix="mt"
        )
        self._load(targets)

    # -------------------------------------------------------------------------

    def _resolve(self, target: str) -> tuple[Path, str | None]:
        """
        Where a target language's model lives, and its selector token if any.

        A dedicated pair model wins over a family model: en-ta beats en-dra for
        Tamil, because a model trained on one pair usually beats one splitting
        its capacity four ways.
        """
        direct = self.model_root / f"{self.source_language}-{target}"
        if (direct / "model.bin").exists():
            return direct, None
        if target in MULTILINGUAL:
            directory, token = MULTILINGUAL[target]
            return self.model_root / directory, token
        return direct, None

    def _load(self, targets: list[str]) -> None:
        resolved = {target: self._resolve(target) for target in targets}

        missing = [t for t, (path, _) in resolved.items() if not (path / "model.bin").exists()]
        if missing:
            wanted = {t: str(resolved[t][0].name) for t in missing}
            raise FileNotFoundError(
                f"no model for {missing} (looked for {wanted} under {self.model_root}). "
                f"Fetch with: python tools/fetch_mt_models.py --pair "
                + " --pair ".join(sorted({resolved[t][0].name for t in missing}))
            )

        # Count sharers per model directory before loading, so a family model
        # gets enough workers for the languages that will hit it concurrently.
        sharers: dict[Path, int] = {}
        for path, _token in resolved.values():
            sharers[path] = sharers.get(path, 0) + 1

        models: dict[Path, _Model] = {}
        for target, (path, token) in resolved.items():
            if path not in models:
                started = time.monotonic()
                models[path] = _Model(path, self.device, self.compute_type, sharers[path])
                log.info(
                    "translation model loaded",
                    extra={
                        "model": path.name,
                        "serves": sharers[path],
                        "device": self.device,
                        "compute_type": self.compute_type,
                        "load_s": round(time.monotonic() - started, 2),
                    },
                )
            self._pairs[target] = _Pair(self.source_language, target, models[path], token)
            log.info(
                "translation pair ready",
                extra={
                    "pair": f"{self.source_language}-{target}",
                    "model": path.name,
                    "target_token": token,
                },
            )

    def supports(self, source: str, target: str) -> bool:
        return source == self.source_language and target in self._pairs

    # -------------------------------------------------------------------------

    async def translate(self, text: str, source: str, target: str) -> str:
        if not self.supports(source, target):
            raise KeyError(f"no loaded pair for {source}->{target}")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._translate_sync, text, target
        )

    async def translate_many(
        self, text: str, source: str, targets: list[str]
    ) -> dict[str, str]:
        """
        Fan one phrase out to every target at once.

        CTranslate2 releases the GIL, so these genuinely run in parallel. One
        language failing must not take the others down: a failed chain returns
        empty text and the session carries on with the rest.
        """
        loop = asyncio.get_running_loop()
        tasks = {
            target: loop.run_in_executor(self._executor, self._translate_sync, text, target)
            for target in targets
            if self.supports(source, target)
        }
        results: dict[str, str] = {}
        for target, task in tasks.items():
            try:
                results[target] = await task
            except Exception:
                log.exception("translation failed", extra={"target": target})
                results[target] = ""
        return results

    def _translate_sync(self, text: str, target: str) -> str:
        return self._pairs[target].translate(text, self.beam_size, self.max_output_ratio)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
