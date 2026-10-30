"""
Clause boundary detection.

A stable prefix is not automatically a good thing to translate. Cutting mid
clause gives the translation model half a thought, and the output shows it.
This module finds the latest defensible cut point in a token run.

Three strengths of boundary:

    STRONG   sentence terminators. Always a good cut.
    MEDIUM   clause separators (comma, semicolon, colon). Usually fine.
    WEAK     coordinating and subordinating conjunctions. Only worth using
             when the alternative is running past max_words, because cutting
             before a conjunction strands the clause it introduces.

Punctuation depends on the STT backend actually producing it. Whisper does.
Some backends do not, and with those only WEAK and the max_words ceiling are
available. Check emits_punctuation at session start rather than discovering it
in the latency numbers.
"""

from __future__ import annotations

import re
from enum import IntEnum

from .stability import normalise_token


class BoundaryStrength(IntEnum):
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3


# Latin, Arabic, Devanagari and CJK sentence terminators.
_STRONG_CHARS = frozenset(".?!۔।॥。？！")
# Clause separators including Arabic comma and semicolon, and CJK forms.
_MEDIUM_CHARS = frozenset(",;:،؛、，；：")

# Tokens whose trailing full stop is an abbreviation, not a sentence end.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt",
        "inc", "ltd", "llc", "co", "corp", "plc",
        "eg", "ie", "etc", "vs", "approx", "dept", "est",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "am", "pm", "no", "vol", "fig", "pp",
    }
)

# English conjunctions. Weak boundaries sit BEFORE these, never after.
_CONJUNCTIONS = frozenset(
    {
        "and", "but", "or", "so", "yet", "for", "nor",
        "because", "although", "though", "while", "whereas", "since",
        "if", "unless", "until", "when", "whenever", "where", "wherever",
        "that", "which", "who", "whom", "whose", "after", "before",
        "however", "therefore", "moreover", "furthermore", "meanwhile",
    }
)

_HAS_DIGIT = re.compile(r"\d")


def _ends_sentence(token: str) -> bool:
    """Whether a trailing full stop really terminates a sentence."""
    if not token or token[-1] not in _STRONG_CHARS:
        return False
    if token[-1] != ".":
        # Question and exclamation marks are unambiguous.
        return True
    stem = token[:-1]
    if not stem:
        return False
    if _HAS_DIGIT.search(stem):
        # "3.5", "1." in a spoken list, version numbers.
        return False
    if len(stem) == 1:
        # Initials: "J. Smith".
        return False
    return normalise_token(stem) not in _ABBREVIATIONS


def token_boundary_strength(token: str) -> BoundaryStrength:
    """Strength of the boundary immediately AFTER this token."""
    if not token:
        return BoundaryStrength.NONE
    if _ends_sentence(token):
        return BoundaryStrength.STRONG
    if token[-1] in _MEDIUM_CHARS:
        return BoundaryStrength.MEDIUM
    return BoundaryStrength.NONE


def is_conjunction(token: str) -> bool:
    return normalise_token(token) in _CONJUNCTIONS


def find_boundary(
    tokens: list[str],
    min_words: int,
    allow_weak: bool = False,
) -> tuple[int, BoundaryStrength]:
    """
    Find the latest usable cut point in `tokens`.

    Returns (exclusive end index, strength). A returned index of 0 means no
    acceptable boundary was found and the caller should keep waiting.

    Scans from the end so the chunk is as long as the available boundaries
    allow. Longer chunks translate better; the max_words ceiling and the wait
    timeout are what stop this running away.
    """
    if len(tokens) < min_words:
        return 0, BoundaryStrength.NONE

    # Strong and medium boundaries, latest first.
    for i in range(len(tokens) - 1, min_words - 2, -1):
        strength = token_boundary_strength(tokens[i])
        if strength >= BoundaryStrength.MEDIUM:
            return i + 1, strength

    if not allow_weak:
        return 0, BoundaryStrength.NONE

    # Weak boundaries sit BEFORE a conjunction, so the conjunction opens the
    # next chunk rather than dangling at the end of this one.
    # The "and" in "one hundred and eighty two" is not a clause break.
    for i in range(len(tokens) - 1, min_words - 1, -1):
        if is_conjunction(tokens[i]) and is_safe_cut(tokens, i):
            return i, BoundaryStrength.WEAK

    return 0, BoundaryStrength.NONE


# Words that cannot end a phrase: the thing they introduce is in the next one.
# Measured on the medical script, 2026-10-28: "an acute myocardial | infarction",
# "within four | and a half hours", "one point | two grams". Each half was
# translated alone, and the Arabic for the second read "half an hour" - a
# thrombolysis window - and "gram twice three times daily" - a dose.
_DANGLING = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "nor",
        "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
        "as", "than", "via", "per", "over", "under", "between", "about",
        "is", "are", "was", "were", "be", "been", "am",
        "my", "your", "our", "their", "its",
        "very", "more", "most", "less", "acute", "chronic",
    }
)

_NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
        "point", "half", "quarter",
    }
)


def _is_number(token: str) -> bool:
    word = normalise_token(token)
    return word in _NUMBER_WORDS or bool(_HAS_DIGIT.search(word))


def is_safe_cut(tokens: list[str], end: int) -> bool:
    """
    Whether cutting after tokens[end - 1] keeps a number and a phrase whole.

    `end` may equal len(tokens): the newest word is the recogniser's frontier,
    and a number there may still be growing ("one hundred" before "and eighty
    two" arrives), so it is not safe either.
    """
    if end <= 0:
        return False
    if token_boundary_strength(tokens[end - 1]) >= BoundaryStrength.MEDIUM:
        return True
    last = normalise_token(tokens[end - 1])
    if last in _DANGLING:
        return False
    if _is_number(tokens[end - 1]):
        if end == len(tokens):
            return False
        nxt = normalise_token(tokens[end])
        # "182 | over 108" and "24 | to 48 hours" are one quantity each.
        if _is_number(tokens[end]) or nxt in ("and", "point", "to", "over", "by", "x"):
            return False
    return True


def safe_cut(tokens: list[str], end: int, min_words: int) -> int:
    """
    The latest cut at or before `end` that is_safe_cut allows, or 0 if none
    leaves at least min_words. A caller that gets 0 should wait for more words,
    up to its own ceiling.
    """
    for i in range(end, min_words - 1, -1):
        if is_safe_cut(tokens, i):
            return i
    return 0
