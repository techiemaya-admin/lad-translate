"""
Feedback on a transcript line, and what the pipeline can honestly learn from it.

The WhatsApp agent's "AI Learnings" turns a thumbs-down plus an expected reply
into text in the LLM's system prompt. That shape is what an operator expects
here, so this keeps it: a thumbs on each line, a "should have been", a switch
to turn a learning off without deleting it.

WHAT IS DIFFERENT, AND WHY IT MUST BE SAID PLAINLY. There is no prompt in this
pipeline. FastConformer, Whisper, Opus-MT and NLLB are fixed neural models;
text cannot change what they produce. What the pipeline DOES have is the
corrections engine - word rules, reloaded live, proven to change the next
phrase within five seconds. So the only learning that is real here is:

    thumbs-down + "should have been"  ->  a word rule  ->  the next phrase

and that is exactly what this derives. Feedback that cannot become a rule - a
whole sentence rephrased, a word added that nothing can be matched against -
is kept as an example and the interface says it changed nothing. Pretending
otherwise would have an operator believe they had fixed something the audience
will hear wrong again in ninety seconds.

WHEN A DIFF BECOMES A RULE. Only when the change is ONE contiguous span,
bounded in length, and the wrong side is something safe to match on. The last
condition is the one that matters: "a" -> "the" is a perfectly sensible edit
to one sentence and a catastrophe as a rule, because it then fires on every
"a" the speaker ever says. Short function words never become rules.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .corrections import Correction, normalise

MAX_SPAN_WORDS = 4
"""
Longest run of words a single rule may replace.

Past this it is a rephrase, not a correction: the operator has rewritten the
sentence, and a rule that matches five specific words in a row will fire once
in the life of the venue while looking, in the list, like it earned its place.
"""

MIN_WRONG_CHARS = 3
"""
Shortest wrong side a rule may have, letters only.

"a", "an", "of", "to", "I" - every one of them is a real word a recogniser can
get wrong, and every one of them as a rule rewrites something the speaker says
every few seconds. The floor is characters rather than a stopword list because
the list would have to exist per language.
"""

FUNCTION_WORDS = frozenset(
    {
        "the", "and", "but", "for", "nor", "yet", "are", "was", "were", "has",
        "had", "not", "you", "our", "his", "her", "its", "who", "how", "why",
        "this", "that", "with", "from", "they", "them", "then", "than", "will",
        "have", "been", "into", "your", "what", "when", "where", "which",
    }
)
"""
English function words above the character floor that are still not safe as a
whole rule. English only, because the source language is: a target-side rule in
Arabic is judged by the character floor alone, which is a known gap and is
said here rather than hidden.
"""


@dataclass(frozen=True, slots=True)
class Derived:
    """What a thumbs-down taught, or why it could not."""

    rule: Correction | None
    reason: str
    """Human sentence. Shown in the panel next to the feedback either way."""

    @property
    def learned(self) -> bool:
        return self.rule is not None


def _tokens(text: str) -> list[str]:
    return normalise(text).split()


def derive_rule(produced: str, expected: str, language: str) -> Derived:
    """
    Turn "this came out as X, it should have been Y" into a rule, if one is
    safe, and into a plain reason if not.

    Compared casefolded, so a change of capitalisation alone is not a
    correction. The rule's wrong side is taken from the PRODUCED text in its
    original casing, because that is the surface the matcher will see again.
    """
    got, want = _tokens(produced), _tokens(expected)
    if not got:
        return Derived(None, "there is nothing in the produced text to match against")
    if not want:
        return Derived(None, "nothing to learn: 'should have been' is empty")
    if [t.casefold() for t in got] == [t.casefold() for t in want]:
        return Derived(None, "the two are the same words; only the casing differs")

    matcher = difflib.SequenceMatcher(
        a=[t.casefold() for t in got], b=[t.casefold() for t in want], autojunk=False
    )
    changes = [op for op in matcher.get_opcodes() if op[0] != "equal"]

    if len(changes) != 1:
        return Derived(
            None,
            f"{len(changes)} separate changes; a rule fixes one phrase at a time, "
            "so this is kept as an example rather than learned",
        )
    tag, i1, i2, j1, j2 = changes[0]

    if tag == "insert":
        return Derived(
            None,
            "words were added and nothing was wrong to match against, so this "
            "cannot become a rule",
        )
    wrong_words = got[i1:i2]
    right_words = want[j1:j2] if tag == "replace" else []

    if len(wrong_words) > MAX_SPAN_WORDS or len(right_words) > MAX_SPAN_WORDS:
        return Derived(
            None,
            f"{max(len(wrong_words), len(right_words))} words changed in a row; "
            "that is a rephrase, kept as an example rather than a rule",
        )

    wrong = " ".join(wrong_words)
    letters = "".join(ch for ch in wrong if ch.isalnum())
    if len(letters) < MIN_WRONG_CHARS:
        return Derived(
            None,
            f"'{wrong}' is too short to be a safe rule: it would fire on every "
            "one the speaker says",
        )
    if len(wrong_words) == 1 and wrong_words[0].casefold() in FUNCTION_WORDS:
        return Derived(
            None,
            f"'{wrong}' is a function word and would fire on every one the "
            "speaker says; kept as an example",
        )

    right = " ".join(right_words)
    verb = "removed" if not right else f"replaced with '{right}'"
    return Derived(
        Correction(wrong, right, language),
        f"'{wrong}' will be {verb} in every phrase from now on",
    )
