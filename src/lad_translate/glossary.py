"""
A domain glossary: this English term is THIS target term, every time.

Corrections (corrections.py) rewrite text on one side of the translator. That
cannot fix a term the translator gets wrong from correct English, because the
wrong output varies with the sentence around it. Measured on the medical test
script, 2026-10-29, translating the script's own text:

    thrombolysis      -> تخثر        "coagulation" - the opposite
    metastasis        -> ورم خبيث   "malignant tumour"
    slurred speech    -> الكلام ملتهب "inflamed speech"
    admitted          -> اعترف       "confessed"

A glossary term is lifted out before translation and put back after:

    "T001 with alteplase is effective"  ->  "T001 مع ألتيبلاس فعال"
                                         ->  "إذابة الخثرة مع ألتيبلاس فعال"

Opus-MT carries an unknown capitalised token straight through, in place, so the
sentence around the term is still translated by the model. If a placeholder
does not survive, the phrase is translated again without the glossary rather
than shipping a sentence with a hole in it or "T001" read aloud.

Matching is the same as corrections: whole words, case-insensitive, longest
first, one pass. List spelling variants as separate entries.

FILE FORMAT. Tab separated, one term per line, `#` comments:

    ar<TAB>thrombolysis<TAB>إذابة الخثرة
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .corrections import normalise

_SLOT = "T{:03d}"
_SLOT_PATTERN = re.compile(r"T(\d{3})")


@dataclass(frozen=True, slots=True)
class Protected:
    text: str
    """The source with each term replaced by a placeholder."""
    slots: tuple[str, ...]
    """Target term for each placeholder, by number."""


class Glossary:
    def __init__(self, entries: list[tuple[str, str, str]]) -> None:
        """`entries` are (target language, source term, target term)."""
        self._compiled: dict[str, tuple[re.Pattern[str], dict[str, str]]] = {}
        by_language: dict[str, dict[str, str]] = {}
        for language, source, target in entries:
            key = normalise(source).casefold()
            if key:
                by_language.setdefault(language, {}).setdefault(key, target)
        for language, lookup in by_language.items():
            alternatives = sorted(lookup, key=len, reverse=True)
            pattern = re.compile(
                r"(?<!\w)(" + "|".join(re.escape(a) for a in alternatives) + r")(?!\w)",
                re.IGNORECASE | re.UNICODE,
            )
            self._compiled[language] = (pattern, lookup)

    @classmethod
    def load(cls, path: Path | str) -> Glossary:
        entries = []
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3 or not all(p.strip() for p in parts):
                raise ValueError(f"{path}:{number}: expected language<TAB>term<TAB>translation")
            entries.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        return cls(entries)

    def languages(self) -> set[str]:
        return set(self._compiled)

    def protect(self, text: str, language: str) -> Protected | None:
        """The text with its terms swapped for placeholders, or None if it has none."""
        compiled = self._compiled.get(language)
        if not compiled or not text:
            return None
        pattern, lookup = compiled
        slots: list[str] = []

        def swap(match: re.Match[str]) -> str:
            slots.append(lookup[normalise(match.group(1)).casefold()])
            return _SLOT.format(len(slots))

        protected = pattern.sub(swap, text)
        if not slots:
            return None
        return Protected(protected, tuple(slots))

    @staticmethod
    def restore(translated: str, protected: Protected) -> str | None:
        """
        Put the target terms back. None if any placeholder went missing or a
        stray one appeared, so the caller can translate without the glossary.
        """
        found = [int(n) for n in _SLOT_PATTERN.findall(translated)]
        if sorted(found) != list(range(1, len(protected.slots) + 1)):
            return None
        return _SLOT_PATTERN.sub(lambda m: protected.slots[int(m.group(1)) - 1], translated)
