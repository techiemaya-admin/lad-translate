"""
NLLB-200 translation through CTranslate2. GPU BACKEND.

One model for 200 languages, against Opus-MT's one model per pair. Two
consequences, one good and one that decides where this can run.

QUALITY. Far better for languages Opus-MT covers only through a family model.
Same sentence, measured:

    Telugu  Opus-MT (en-dra)  సెంటర్ అవతల పదకొండు శాతం పెరిగింది
                              "center" transliterated, "revenue" dropped
            NLLB              ఈ రంగం లో ఆదాయం గత సంవత్సరం 11 శాతం పెరిగింది
                              correct words throughout

    Hindi   Opus-MT (en-hi)   सेक्टर पार पर Ruue पिछले साल 11 प्रतिशत बढ़ा
                              untranslated Latin garbage
            NLLB              इस क्षेत्र में राजस्व में पिछले साल 11 प्रतिशत की वृद्धि हुई है

For European languages the two are comparable, so the gain is concentrated
exactly where Opus-MT is weakest.

COST. One model means the whole fan-out is a single batched call rather than N
model invocations, which is the efficient shape. It is still far heavier:

    Opus-MT, 4 languages, CPU     ~300ms
    NLLB-600M, 5 languages, CPU   ~4500ms

15x slower, against a 2 second end to end budget. This is a GPU backend. On
CPU it is for quality comparison only, and the pipeline will shed audio behind
it. Use adapters/mt_opus.py on the dev Mac.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..obs.log import get_logger
from .base import MtAdapter

log = get_logger(__name__)

DEFAULT_MODEL_PATH = Path("models/mt/nllb-600m")

# BCP-47 to FLORES-200. NLLB names a language AND its script, so the same
# language in two scripts is two codes: Urdu is urd_Arab, not urd_Latn.
FLORES: dict[str, str] = {
    "ar": "arb_Arab",
    "bn": "ben_Beng",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fa": "pes_Arab",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "nl": "nld_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


class NllbMtAdapter(MtAdapter):
    """One model, every language, fanned out in a single batched call."""

    name = "nllb-200"

    def __init__(
        self,
        source_language: str,
        targets: list[str],
        model_path: Path | str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        compute_type: str | None = None,
        beam_size: int = 1,
        max_output_ratio: float = 3.0,
    ) -> None:
        import ctranslate2
        from tokenizers import Tokenizer

        self.model_path = Path(model_path)
        if not (self.model_path / "model.bin").exists():
            raise FileNotFoundError(
                f"no NLLB model at {self.model_path}. Fetch it with: "
                "python tools/fetch_mt_models.py --nllb"
            )

        self.source_language = source_language
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.beam_size = beam_size
        self.max_output_ratio = max_output_ratio

        unknown = [t for t in [source_language, *targets] if t not in FLORES]
        if unknown:
            raise KeyError(
                f"no FLORES-200 code for {unknown}; add it to FLORES in mt_nllb.py"
            )
        self._source_code = FLORES[source_language]
        self._targets = {t: FLORES[t] for t in targets}

        started = time.monotonic()
        self._tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self._vocab = self._tokenizer.get_vocab()
        self._translator = ctranslate2.Translator(
            str(self.model_path),
            device=device,
            compute_type=self.compute_type,
            inter_threads=1,
            # One model serves every language, so give it the cores rather than
            # splitting them across per-pair models as Opus-MT does.
            intra_threads=4,
        )
        # A single executor slot: the model is invoked once per chunk with the
        # whole batch, so there is nothing to run concurrently.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nllb")
        log.info(
            "NLLB model loaded",
            extra={
                "device": device,
                "compute_type": self.compute_type,
                "languages": sorted(self._targets),
                "load_s": round(time.monotonic() - started, 2),
                "latency_credible": device == "cuda",
            },
        )
        if device != "cuda":
            log.warning(
                "NLLB on CPU is roughly 15x slower than Opus-MT and cannot meet "
                "the latency budget; use it for quality comparison only"
            )

    # -------------------------------------------------------------------------

    def supports(self, source: str, target: str) -> bool:
        return source == self.source_language and target in self._targets

    def _encode(self, text: str) -> list[str]:
        """
        NLLB source form: source language token, the text, end of sentence.

        Special tokens from the tokenizer are stripped first; NLLB wants its
        own language token in that position, not the tokenizer's <s>.
        """
        pieces = self._tokenizer.encode(text).tokens
        body = [t for t in pieces if t not in ("<s>", "</s>")]
        return [self._source_code, *body, "</s>"]

    def _decode(self, hypothesis: list[str], target_code: str) -> str:
        ids = [
            self._vocab[t]
            for t in hypothesis
            if t in self._vocab and t != target_code
        ]
        return self._tokenizer.decode(ids)

    # -------------------------------------------------------------------------

    async def translate(self, text: str, source: str, target: str) -> str:
        result = await self.translate_many(text, source, [target])
        return result.get(target, "")

    async def translate_many(
        self, text: str, source: str, targets: list[str]
    ) -> dict[str, str]:
        """
        Fan out in ONE call.

        This is the whole reason to use a single multilingual model: the source
        is encoded once and the batch decodes every target together, rather
        than N models each doing their own encode.
        """
        wanted = [t for t in targets if self.supports(source, t)]
        if not wanted or not text.strip():
            return {t: "" for t in wanted}
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._translate_sync, text, wanted
        )

    def _translate_sync(self, text: str, targets: list[str]) -> dict[str, str]:
        source = self._encode(text)
        codes = [self._targets[t] for t in targets]
        max_len = max(32, int(len(source) * self.max_output_ratio))
        try:
            results = self._translator.translate_batch(
                [source] * len(codes),
                target_prefix=[[code] for code in codes],
                beam_size=self.beam_size,
                max_decoding_length=max_len,
            )
        except Exception:
            # One bad batch must not silence the room for the rest of the event.
            log.exception("NLLB batch failed", extra={"targets": targets})
            return {t: "" for t in targets}
        return {
            target: self._decode(result.hypotheses[0], code)
            for target, code, result in zip(targets, codes, results)
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False)
