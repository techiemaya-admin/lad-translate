"""
Per-language translation routing.

Neither backend is right for every language, so pick per language.

MEASURED, same transcript through both, JFK excerpt:

    French   near identical. Opus-MT is 15x faster, so it wins on cost.
    Hindi    Opus-MT rendered "revolutionary beliefs" as मूलतत्त्ववादी,
             which means FUNDAMENTALIST. Grammatical, confident, and the
             opposite of what was said. It also emitted "Ruue" for "Revenue".
    Telugu   Opus-MT hallucinated on short input:
                 "the hand of God."  -> దేవుని చేతి. வெறுமென ఒక రాత్రి,
                                        ఒక రాత్రి, ఒక రాత్రి, ఒక నగలను...
             Four words in, fifteen words of Tamil-laced nonsense out. The
             spoken output ran 45.0s against NLLB's 38.1s for the same source:
             seven extra seconds of an audience being read garbage.

The failures cluster on SHORT input, and the phrase chunker produces short
chunks by design. Opus-MT's family models fail precisely on the shape this
architecture generates, which is why Telugu is not merely weaker but unsafe.

So: Indic languages to NLLB, everything else to Opus-MT. Fast where fast is
good enough, correct where it is not.
"""

from __future__ import annotations

import asyncio

from ..obs.log import get_logger
from .base import MtAdapter

log = get_logger(__name__)

OPUS = "opus-mt"
NLLB = "nllb-200"

DEFAULT_ROUTES: dict[str, str] = {
    # Measured failures. Both were bad enough to put in front of nobody.
    "hi": NLLB,
    "te": NLLB,
    # Served by the same en-dra family model as Telugu. Not individually
    # measured, but the failure mode belongs to the model rather than to
    # Telugu, so routing them to Opus-MT would be assuming the best about a
    # model already caught hallucinating.
    "ta": NLLB,
    "ml": NLLB,
    "kn": NLLB,
    # No dedicated Opus-MT pair fetched for these either.
    "bn": NLLB,
    "mr": NLLB,
    "ur": NLLB,
}
"""
Language to backend. Anything absent goes to Opus-MT.

Opus-MT is the default because it is 15x faster and, where it is good, it is as
good. Entries here are exceptions earned by evidence, not a preference for the
bigger model.
"""

DEFAULT_BACKEND = OPUS


def route_for(language: str, routes: dict[str, str] | None = None) -> str:
    """
    Which backend serves this language.

    `routes is None` means "use the defaults"; an empty dict means "no
    exceptions, send everything to the default backend". Writing this as
    `routes or DEFAULT_ROUTES` conflates the two, so passing {} to force
    everything onto Opus-MT would silently get the Indic routing instead.
    """
    table = DEFAULT_ROUTES if routes is None else routes
    return table.get(language, DEFAULT_BACKEND)


class RoutingMtAdapter(MtAdapter):
    """Sends each language to the backend that handles it best."""

    name = "routing"

    def __init__(
        self,
        source_language: str,
        targets: list[str],
        routes: dict[str, str] | None = None,
        device: str = "cpu",
        opus_options: dict | None = None,
        nllb_options: dict | None = None,
    ) -> None:
        self.source_language = source_language
        self.routes = DEFAULT_ROUTES if routes is None else routes
        self._backends: dict[str, MtAdapter] = {}
        self._assignment: dict[str, str] = {
            target: route_for(target, self.routes) for target in targets
        }

        grouped: dict[str, list[str]] = {}
        for target, backend in self._assignment.items():
            grouped.setdefault(backend, []).append(target)

        for backend, languages in grouped.items():
            if backend == OPUS:
                from .mt_opus import OpusMtAdapter

                self._backends[backend] = OpusMtAdapter(
                    source_language, languages, device=device, **(opus_options or {})
                )
            elif backend == NLLB:
                from .mt_nllb import NllbMtAdapter

                self._backends[backend] = NllbMtAdapter(
                    source_language, languages, device=device, **(nllb_options or {})
                )
            else:
                raise KeyError(f"unknown translation backend {backend!r}")

        log.info(
            "translation routing resolved",
            extra={
                "assignment": self._assignment,
                "backends": sorted(grouped),
                "device": device,
            },
        )
        if device != "cuda" and NLLB in grouped:
            log.warning(
                "NLLB is routed on CPU and is roughly 15x slower than Opus-MT; "
                "the pipeline will shed audio behind it",
                extra={"nllb_languages": sorted(grouped[NLLB])},
            )

    # -------------------------------------------------------------------------

    def backend_for(self, language: str) -> str:
        return self._assignment.get(language, DEFAULT_BACKEND)

    def supports(self, source: str, target: str) -> bool:
        if source != self.source_language or target not in self._assignment:
            return False
        backend = self._backends.get(self._assignment[target])
        return backend is not None and backend.supports(source, target)

    async def translate(self, text: str, source: str, target: str) -> str:
        result = await self.translate_many(text, source, [target])
        return result.get(target, "")

    async def translate_many(
        self, text: str, source: str, targets: list[str]
    ) -> dict[str, str]:
        """
        Fan out across backends concurrently, batching within each.

        Each backend gets one call with its whole share, so NLLB still decodes
        its languages in a single batch rather than one at a time. The two
        backends then run in parallel, so the slower one sets the pace instead
        of the sum of both.
        """
        wanted = [t for t in targets if self.supports(source, t)]
        if not wanted:
            return {}

        grouped: dict[str, list[str]] = {}
        for target in wanted:
            grouped.setdefault(self._assignment[target], []).append(target)

        async def run(backend_name: str, languages: list[str]) -> dict[str, str]:
            try:
                return await self._backends[backend_name].translate_many(
                    text, source, languages
                )
            except Exception:
                # One backend failing must not silence the languages served by
                # the other.
                log.exception(
                    "translation backend failed",
                    extra={"backend": backend_name, "languages": languages},
                )
                return {lang: "" for lang in languages}

        parts = await asyncio.gather(
            *(run(name, langs) for name, langs in grouped.items())
        )
        merged: dict[str, str] = {}
        for part in parts:
            merged.update(part)
        return merged

    def close(self) -> None:
        for backend in self._backends.values():
            if hasattr(backend, "close"):
                backend.close()
