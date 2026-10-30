"""
Spoken form to written form, between recognition and translation.

FastConformer writes what it hears: "thirty eight point nine", "e c g". The
translator was trained on written text, and it does very differently with the
two forms. Measured on the medical test script, 2026-10-29:

    spoken form (as recognised)                 written form (same words)
    febrile at thirty eight point nine degrees  38.9 درجة مئوية
      -> "in February 38 °C nine °C"
    one point two grams three times daily       1.2 غرام ثلاث مرات يوميا
      -> "point one / gram twice three times"
    e c g                                       ECG kept, or the right term
      -> ه ج ز (three Arabic letters, unreadable)

So the fix is not in the translator: hand it the written form.

WHAT CONVERTS, AND WHY THE LINE IS WHERE IT IS.

  A number becomes digits when it is 10 or more, has a decimal part, is
  followed by a unit ("five milligrams", "two litres"), or follows a word that
  numbers things ("bed four", "type two", "stage three"). Below ten with
  none of those it stays a word: "on one side", "two thirds", "one of the
  greatest" are not quantities, and "on 1 side" translates worse, not better.

  A run of two or more single letters becomes an acronym: "m r s a" -> MRSA,
  "c t scan" -> CT scan. A leading "a" or "i" is an article or a pronoun, not
  part of it: "a c t scan" -> "a CT scan". A digit word between letters joins
  them: "h b a one c" -> HBA1C.

Nothing here knows about medicine. It is the written form of English, which
any venue's translation benefits from.
"""

from __future__ import annotations

import re

_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}

UNITS = frozenset(
    {
        "percent", "per", "degrees", "degree", "grams", "gram", "milligrams",
        "milligram", "micrograms", "kilograms", "kilos", "litres", "litre",
        "liters", "liter", "millilitres", "milliliters", "millimoles", "millimole",
        "units", "unit", "hours", "hour", "minutes", "minute", "seconds", "second",
        "days", "day", "weeks", "week", "months", "month", "years", "year",
        "beats", "breaths", "times", "doses", "dose", "tablets", "tablet",
        "millimetres", "millimeters", "centimetres", "centimeters", "metres",
        "meters", "kilometres", "kilometers", "mg", "ml", "kg", "mmol",
    }
)

COUNTING_WORDS = frozenset(
    {"bed", "type", "stage", "grade", "class", "level", "phase", "number",
     "room", "step", "ph", "page", "line", "chapter", "section", "ward"}
)

_FUNCTION_WORDS = frozenset(
    {"and", "or", "but", "in", "on", "at", "of", "to", "for", "with", "is",
     "are", "was", "were", "as", "by", "from", "than", "that", "which"}
)

_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _split(token: str) -> tuple[str, str, str]:
    """Leading punctuation, the word, trailing punctuation."""
    m = _EDGE.match(token)
    assert m is not None
    return m.group(1), m.group(2), m.group(3)


def _below_hundred(words: list[str], i: int) -> tuple[int, int] | None:
    """Parse 0-99 at words[i]. Returns (value, next index) or None."""
    if i >= len(words):
        return None
    w = words[i]
    if w in _TENS:
        if i + 1 < len(words) and words[i + 1] in _ONES and words[i + 1] not in ("zero", "oh"):
            return _TENS[w] + _ONES[words[i + 1]], i + 2
        return _TENS[w], i + 1
    if w in _TEENS:
        return _TEENS[w], i + 1
    if w in _ONES and w != "oh":
        return _ONES[w], i + 1
    return None


def _integer(words: list[str], i: int) -> tuple[int, int] | None:
    """Parse an integer up to 999,999 at words[i]."""
    first = _below_hundred(words, i)
    if first is None:
        return None
    value, j = first
    total = 0
    while j < len(words) and words[j] in _SCALES:
        scale = _SCALES[words[j]]
        value *= scale
        j += 1
        if scale == 1000:
            total += value
            value = 0
        # "one hundred AND eighty two", but never "sixty and my mother".
        k = j + 1 if j < len(words) and words[j] == "and" else j
        rest = _below_hundred(words, k)
        if rest is not None and (k == j or rest[0] > 0):
            more, after = rest
            if after < len(words) and words[after] in _SCALES:
                value, j = more, after
                continue
            value += more
            j = after
        if j >= len(words) or words[j] not in _SCALES:
            break
    return total + value, j


def _number(words: list[str], i: int) -> tuple[str, int, bool] | None:
    """
    A number starting at words[i]: (written form, next index, is_decimal).
    """
    parsed = _integer(words, i)
    if parsed is None:
        return None
    value, j = parsed
    # "thirty eight point nine", "one point two"
    if j + 1 < len(words) and words[j] == "point" and words[j + 1] in _ONES:
        digits = []
        k = j + 1
        while k < len(words) and words[k] in _ONES:
            digits.append(str(_ONES[words[k]]))
            k += 1
        return f"{value}.{''.join(digits)}", k, True
    # "four and a half hours"
    if words[j:j + 3] == ["and", "a", "half"]:
        return f"{value}.5", j + 3, True
    return str(value), j, False


def _numbers(tokens: list[str]) -> list[str]:
    words = [_split(t)[1].lower() for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        found = _number(words, i)
        if found is None:
            out.append(tokens[i])
            i += 1
            continue
        written, j, decimal = found
        # Punctuation inside the span would mean two separate numbers, as in
        # "five, six" - leave those alone rather than guess.
        if any(_split(t)[2] for t in tokens[i:j - 1]) or any(_split(t)[0] for t in tokens[i + 1:j]):
            out.append(tokens[i])
            i += 1
            continue
        value = float(written)
        before = words[i - 1] if i > 0 else ""
        after = words[j] if j < len(words) else ""
        convert = value >= 10 or decimal or after in UNITS or before in COUNTING_WORDS
        if not convert:
            out.extend(tokens[i:j])
        else:
            lead, _, _ = _split(tokens[i])
            _, _, trail = _split(tokens[j - 1])
            out.append(f"{lead}{written}{trail}")
        i = j
    return out


def _is_letter(word: str) -> bool:
    return len(word) == 1 and word.isalpha()


def _acronyms(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        _, word, trail = _split(tokens[i])
        if not _is_letter(word) or trail:
            out.append(tokens[i])
            i += 1
            continue
        # Collect a run of single letters, with a digit word allowed between
        # two letters ("h b a one c").
        j = i
        parts: list[str] = []
        while j < len(tokens):
            _, w, t = _split(tokens[j])
            if _is_letter(w):
                parts.append(w.upper())
                j += 1
            elif (w.lower() in _ONES and parts and j + 1 < len(tokens)
                  and _is_letter(_split(tokens[j + 1])[1])):
                parts.append(str(_ONES[w.lower()]))
                j += 1
            else:
                break
            if t:
                break
        # "a c t scan": the article is not part of the acronym.
        start = i
        if len(parts) > 2 and parts[0] in ("A", "I"):
            out.append(tokens[i])
            parts = parts[1:]
            start = i + 1
        # And a trailing one is an article when a content word follows it:
        # "h b a one c a lipid profile", but "m r s a and carbapenem".
        if (len(parts) > 2 and parts[-1] == "A" and j < len(tokens)
                and _split(tokens[j])[1].lower() not in _FUNCTION_WORDS):
            parts = parts[:-1]
            j -= 1
        if len(parts) >= 2:
            _, _, trail = _split(tokens[j - 1])
            out.append(f"{_split(tokens[start])[0]}{''.join(parts)}{trail}")
            i = j
        else:
            out.extend(tokens[start:j] if j > start else [tokens[i]])
            i = max(j, i + 1)
    return out


def spoken_to_written(text: str) -> str:
    """Numbers to digits and spelled letters to acronyms, as described above."""
    if not text:
        return text
    tokens = text.split()
    return " ".join(_numbers(_acronyms(tokens)))
