"""
Prefix stability tracking (LocalAgreement-n).

Streaming STT backends revise their interim output. A backend may report
"we are going to", then "we are going to do buy", then "we're going to Dubai".
Committing the second of those means the audience hears nonsense, and there is
no way to unsay it.

LocalAgreement-n treats a prefix as stable once the last n consecutive
hypotheses all agree on it. Raising n buys accuracy and costs latency, and that
trade is the chunker's central knob.

Comparison is done on normalised tokens so that punctuation and casing churn
(which these backends do constantly) does not reset stability. The text handed
back for translation comes from the most recent hypothesis, because that one
has the best punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from dataclasses import dataclass

from ..adapters.base import Hypothesis

# Strip leading and trailing punctuation for comparison only. Intra-word marks
# (apostrophes, hyphens) stay, because "we're" and "were" are different words.
_EDGE_PUNCT = re.compile(r"^[^\w؀-ۿऀ-ॿ]+|[^\w؀-ۿऀ-ॿ]+$")


def normalise_token(token: str) -> str:
    """Lowercase and strip edge punctuation, for the agreement comparison only."""
    stripped = _EDGE_PUNCT.sub("", token)
    return unicodedata.normalize("NFKC", stripped).casefold()


def tokenise(text: str) -> list[str]:
    """Split into whitespace-delimited tokens, punctuation left attached."""
    return text.split()


@dataclass(slots=True)
class StablePrefix:
    """The portion of the transcript that the last n hypotheses agree on."""

    tokens: list[str]
    """Surface tokens from the most recent hypothesis, punctuation intact."""

    first_seen_wall: float
    """
    time.monotonic() when this prefix length was first reached.

    The chunker subtracts this from commit time to get stability_lag, which
    isolates the chunker's own share of the latency budget.
    """

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


class StabilityTracker:
    """
    Rolling LocalAgreement-n over incoming hypotheses.

    Feed every hypothesis in. Ask for `stable_prefix()` after each one.
    """

    def __init__(self, agreement_n: int = 2) -> None:
        if agreement_n < 1:
            raise ValueError("agreement_n must be at least 1")
        self._n = agreement_n
        self._window: deque[list[str]] = deque(maxlen=agreement_n)
        self._latest_surface: list[str] = []
        self._prefix_first_seen: dict[int, float] = {}
        self._committed = 0
        self._contradicted = False

    @property
    def committed_count(self) -> int:
        """How many tokens have been handed to the chunker and cannot be recalled."""
        return self._committed

    @property
    def saw_contradiction(self) -> bool:
        """
        True if a hypothesis disagreed with tokens already committed.

        The audience has already heard those words. This is the accuracy cost
        of the current agreement_n, and it should be reported alongside any
        latency figure rather than buried.
        """
        return self._contradicted

    def reset_contradiction(self) -> bool:
        was = self._contradicted
        self._contradicted = False
        return was

    def add(self, hyp: Hypothesis) -> None:
        """Record a hypothesis. Finals are treated as agreeing with themselves."""
        surface = tokenise(hyp.text)
        normalised = [normalise_token(t) for t in surface]

        self._detect_contradiction(normalised)
        self._latest_surface = surface

        if hyp.is_final:
            # A final settles the matter, so fill the window with it rather
            # than waiting n more hypotheses that will never arrive.
            for _ in range(self._n):
                self._window.append(normalised)
        else:
            self._window.append(normalised)

        self._record_first_seen(hyp.t_wall)

    def _detect_contradiction(self, normalised: list[str]) -> None:
        """Check whether this hypothesis disagrees with already-committed tokens."""
        if self._committed == 0:
            return
        if len(normalised) < self._committed:
            # Backend shortened its transcript below what we already spoke.
            self._contradicted = True
            return
        previous = list(self._window[-1]) if self._window else []
        if len(previous) < self._committed:
            return
        if normalised[: self._committed] != previous[: self._committed]:
            self._contradicted = True

    def _record_first_seen(self, t_wall: float) -> None:
        length = self._agreed_length()
        # Only the first sighting counts, so a prefix that stalls accumulates lag.
        self._prefix_first_seen.setdefault(length, t_wall)

    def _agreed_length(self) -> int:
        """Length of the longest token prefix common to every hypothesis in the window."""
        if len(self._window) < self._n:
            return 0
        shortest = min(len(h) for h in self._window)
        agreed = 0
        for i in range(shortest):
            token = self._window[0][i]
            if all(h[i] == token for h in self._window):
                agreed = i + 1
            else:
                break
        return agreed

    def stable_prefix(self) -> StablePrefix | None:
        """
        The agreed prefix beyond what has already been committed.

        Returns None when nothing new is stable yet.
        """
        agreed = self._agreed_length()
        if agreed <= self._committed:
            return None
        # Guard against the surface list being shorter than the agreed length,
        # which happens when a final arrives with fewer tokens than the window.
        end = min(agreed, len(self._latest_surface))
        if end <= self._committed:
            return None
        tokens = self._latest_surface[self._committed : end]
        if not tokens:
            return None
        return StablePrefix(
            tokens=tokens,
            first_seen_wall=self._prefix_first_seen.get(agreed, 0.0),
        )

    def commit(self, token_count: int) -> None:
        """Mark tokens as emitted. They can never be revised after this."""
        self._committed += token_count

    def pending_after_commit(self) -> list[str]:
        """Uncommitted surface tokens, stable or not. Used to flush at session end."""
        return self._latest_surface[self._committed :]
