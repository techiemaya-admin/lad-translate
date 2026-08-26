"""Types and tuning knobs for the phrase chunker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommitReason(str, Enum):
    """
    Why a chunk was emitted.

    Tune on the distribution of these, not on latency alone. A healthy session
    is mostly CLAUSE. Heavy MAX_WORDS or STABILITY_TIMEOUT means the chunker is
    being forced out rather than finding natural boundaries, which shows up as
    clumsy translation long before it shows up in the latency numbers.
    """

    CLAUSE = "clause"
    """Stable prefix ended at a punctuation boundary. The good case."""

    MAX_WORDS = "max_words"
    """Stable prefix grew too long to hold. Speaker is not pausing."""

    STABILITY_TIMEOUT = "stability_timeout"
    """Prefix was stable but never reached a boundary before max_wait_s."""

    SILENCE = "silence"
    """Audio went quiet. Flush whatever is pending."""

    FINAL = "final"
    """Backend marked the hypothesis final, or the session ended."""


@dataclass(frozen=True, slots=True)
class PhraseChunk:
    """A unit of text committed for translation. Once emitted it cannot be recalled."""

    chunk_id: int
    text: str
    t_audio_start: float
    t_audio_end: float
    """Audio position of the chunk's last word. This is the clock reference for latency."""

    t_wall_committed: float
    reason: CommitReason
    word_count: int

    stability_lag: float
    """
    Seconds between this text first appearing in a hypothesis and it being
    committed. This is the chunker's own contribution to the latency budget,
    isolated from STT, translation and TTS. It is the number to optimise.
    """

    revised_after_commit: bool = False
    """
    Set by the chunker if a later hypothesis contradicted text already emitted.
    The audience has by then heard the wrong words. Count these: the rate is
    the real accuracy cost of committing early, and it is the other half of
    every latency decision made here.
    """


@dataclass(slots=True)
class ChunkerConfig:
    """
    Tuning for the phrase chunker.

    Every default here is a starting point, not a validated value. They are
    meant to be swept by tools/chunker_replay.py against real conference audio.
    """

    agreement_n: int = 2
    """
    How many consecutive hypotheses must agree on a prefix before it counts as
    stable (LocalAgreement-n). 2 is the usual choice. 3 is safer and slower.
    """

    min_words: int = 4
    """
    Do not emit shorter than this. Short fragments translate badly and produce
    choppy audio, and the per-chunk TTS overhead stops being worth it.
    """

    max_words: int = 25
    """Force a commit above this, even mid-clause. Bounds worst case latency."""

    max_wait_s: float = 1.0
    """
    Longest a stable prefix may wait for a clause boundary. This is a direct
    latency ceiling for the chunker stage: raise it for better boundaries,
    lower it to cut lag.
    """

    silence_commit_s: float = 0.6
    """Flush pending text after this much audio with no new hypothesis."""

    allow_conjunction_boundary: bool = True
    """
    Permit weak boundaries at conjunctions when close to max_words. Improves
    phrasing when a speaker runs on, at some cost to translation quality
    because the clause is cut early.
    """

    conjunction_threshold: float = 0.7
    """Fraction of max_words above which weak boundaries become acceptable."""
