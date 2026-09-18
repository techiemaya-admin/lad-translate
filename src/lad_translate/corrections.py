"""
Operator corrections: "that word is wrong, here is the right one".

A recogniser gets a name wrong the SAME way every time. "Irene Adler" heard as
"I can Adler" is not a one-off to be tidied out of the archive afterwards - it
will land again in ninety seconds, and the audience is listening now. So a
correction here is a RULE that changes what happens next, and only incidentally
something that can be replayed over what already happened.

TWO PLACES A RULE CAN APPLY, and the difference decides which one to write.

  source    language == the session's source. Applied to the phrase BEFORE it
            reaches translation, so every target language is fixed at once and
            the stored transcript reads correctly. This is where a misheard
            name belongs.

  target    language == one target code. Applied to that language's text after
            translation. For when the English is right and one language gets
            it wrong - a brand name that French insists on translating.

MATCHING, and why each choice is what it is.

  Whole words only. A rule "ad" -> "AD" must not turn "Adler" into "ADler".
  Boundaries are (?<!\\w) and (?!\\w) rather than \\b, because a phrase may
  begin or end with punctuation and \\b flips meaning when it does.

  Case-insensitive match, verbatim replacement. A speaker says a name at the
  start of a sentence and in the middle of one; the operator should write the
  casing they want ONCE and get it both times.

  Longest rule first, so "Irene Adler" beats a separate "Irene".

  ONE PASS. Every rule is tried at each position exactly once, so a
  replacement can never be re-matched by another rule. Without this, the pair
  "colour" -> "color" and "color" -> "colour" is an infinite loop, and an
  operator will eventually write that pair.

  Literal text, never a pattern. Whatever the operator typed is escaped: they
  are correcting words, not writing regexes, and "C++" should mean "C++".

WHAT THIS DOES NOT DO. It does not re-synthesise audio the audience has
already heard. A correction applied mid-session fixes every phrase from that
moment on; the ones already spoken are gone, and the retroactive pass only
rewrites the stored transcript so the record and the downloads are right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_RULES = 500
"""
Enough for any event's names and terms, and a bound on the regex this builds.

An unbounded alternation compiled from user text is a way to make a session
pause for a quarter of a second on every phrase; the console refuses past this
rather than letting a glossary grow until someone notices the latency.
"""


@dataclass(frozen=True, slots=True)
class Correction:
    """One rule. `language` is the source code, or one target code."""

    wrong: str
    right: str
    language: str

    def __post_init__(self) -> None:
        if not self.wrong.strip():
            raise ValueError("the text to correct cannot be blank")
        if not self.language.strip():
            raise ValueError("a correction needs a language")


def normalise(text: str) -> str:
    """Collapse whitespace, so a pasted phrase matches a spoken one."""
    return " ".join(text.split())


class Corrections:
    """
    The rules for one session, compiled once per language.

    Built fresh whenever the set changes rather than mutated, so a session
    reading it never sees half an update.
    """

    def __init__(self, rules: list[Correction]) -> None:
        self._by_language: dict[str, list[Correction]] = {}
        for rule in rules[:MAX_RULES]:
            self._by_language.setdefault(rule.language, []).append(rule)
        self._compiled: dict[str, tuple[re.Pattern[str], dict[str, str]]] = {}
        for language, group in self._by_language.items():
            compiled = self._compile(group)
            if compiled is not None:
                self._compiled[language] = compiled

    @staticmethod
    def _compile(group: list[Correction]):
        lookup: dict[str, str] = {}
        for rule in group:
            key = normalise(rule.wrong)
            # First rule wins on a duplicate, so the list's order is the
            # operator's precedence rather than dictionary chance.
            lookup.setdefault(key.casefold(), rule.right)
        if not lookup:
            return None
        # Longest first: at any position the regex takes the first alternative
        # that matches, so "Irene Adler" has to be offered before "Irene".
        alternatives = sorted(lookup, key=len, reverse=True)
        pattern = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(a) for a in alternatives) + r")(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
        return pattern, lookup

    def languages(self) -> set[str]:
        return set(self._compiled)

    def apply(self, text: str, language: str) -> tuple[str, int]:
        """
        Correct `text` for one language. Returns the text and how many words
        were changed, because a rule that never fires is worth showing to the
        operator who wrote it.
        """
        compiled = self._compiled.get(language)
        if not compiled or not text:
            return text, 0
        pattern, lookup = compiled
        hits = 0

        def swap(match: re.Match[str]) -> str:
            nonlocal hits
            replacement = lookup.get(normalise(match.group(1)).casefold())
            if replacement is None:
                return match.group(0)
            hits += 1
            return replacement

        return pattern.sub(swap, text), hits

    def __len__(self) -> int:
        return sum(len(g) for g in self._by_language.values())

    def __bool__(self) -> bool:
        return bool(self._compiled)


EMPTY = Corrections([])
