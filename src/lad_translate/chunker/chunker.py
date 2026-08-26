"""
The phrase chunker.

Turns a stream of revisable STT hypotheses into a stream of committed phrases
for translation. This is the component that sets the floor on end to end
latency, and the one place where latency is traded directly against accuracy.

The API is push based on purpose:

    chunks = chunker.feed(hypothesis)   # after each hypothesis
    chunks = chunker.tick(now)          # when audio is quiet
    chunks = chunker.flush()            # at end of session

No async, no I/O, no model. That keeps it testable against recorded transcripts
with no GPU and no network, which is how it should be tuned. See
tools/chunker_replay.py.
"""

from __future__ import annotations

from ..adapters.base import Hypothesis
from .clause import find_boundary
from .stability import StabilityTracker
from .types import ChunkerConfig, CommitReason, PhraseChunk


class PhraseChunker:
    """Commits stable, clause-aligned phrases from a revisable transcript stream."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._tracker = StabilityTracker(self.config.agreement_n)
        self._next_id = 0
        self._last_hyp_wall: float | None = None
        self._last_hyp: Hypothesis | None = None
        self._audio_cursor = 0.0
        """Audio position of the last committed word. Start of the next chunk."""

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def feed(self, hyp: Hypothesis) -> list[PhraseChunk]:
        """Take one hypothesis. Return any phrases that became committable."""
        self._tracker.add(hyp)
        self._last_hyp_wall = hyp.t_wall
        self._last_hyp = hyp

        if hyp.is_final:
            return self._emit_all(hyp, CommitReason.FINAL)
        return self._try_emit(hyp)

    def tick(self, now: float) -> list[PhraseChunk]:
        """
        Call when no hypothesis has arrived for a while.

        Silence is a boundary the STT backend will not tell us about, so the
        chunker has to notice it itself. Without this, a speaker who stops mid
        sentence leaves the last phrase stuck in the buffer until they resume.
        """
        if self._last_hyp_wall is None or self._last_hyp is None:
            return []
        if now - self._last_hyp_wall < self.config.silence_commit_s:
            return []
        return self._emit_all(self._last_hyp, CommitReason.SILENCE, now=now)

    def flush(self) -> list[PhraseChunk]:
        """Emit everything outstanding. Called at end of session."""
        if self._last_hyp is None:
            return []
        return self._emit_all(self._last_hyp, CommitReason.FINAL)

    # -------------------------------------------------------------------------
    # Commit decisions
    # -------------------------------------------------------------------------

    def _try_emit(self, hyp: Hypothesis) -> list[PhraseChunk]:
        prefix = self._tracker.stable_prefix()
        if prefix is None:
            return []

        cfg = self.config
        tokens = prefix.tokens

        # Hard ceiling. The speaker is not pausing and we cannot hold longer.
        if len(tokens) >= cfg.max_words:
            return [self._commit(hyp, prefix, cfg.max_words, CommitReason.MAX_WORDS)]

        # Weak boundaries only become acceptable near the ceiling, because
        # cutting before a conjunction costs translation quality.
        allow_weak = cfg.allow_conjunction_boundary and (
            len(tokens) >= cfg.conjunction_threshold * cfg.max_words
        )
        end, _strength = find_boundary(tokens, cfg.min_words, allow_weak=allow_weak)
        if end > 0:
            return [self._commit(hyp, prefix, end, CommitReason.CLAUSE)]

        # Stable but no boundary in sight. Hold, up to the latency ceiling.
        lag = hyp.t_wall - prefix.first_seen_wall
        if lag >= cfg.max_wait_s and len(tokens) >= cfg.min_words:
            return [self._commit(hyp, prefix, len(tokens), CommitReason.STABILITY_TIMEOUT)]

        return []

    def _emit_all(
        self, hyp: Hypothesis, reason: CommitReason, now: float | None = None
    ) -> list[PhraseChunk]:
        """
        Drain everything outstanding, respecting max_words.

        Used for finals, silence and session end, where holding back for a
        better boundary no longer buys anything.
        """
        chunks: list[PhraseChunk] = []
        while True:
            prefix = self._tracker.stable_prefix()
            if prefix is None or not prefix.tokens:
                break
            take = min(len(prefix.tokens), self.config.max_words)
            chunks.append(self._commit(hyp, prefix, take, reason, now=now))
        return chunks

    def _commit(
        self,
        hyp: Hypothesis,
        prefix,
        take: int,
        reason: CommitReason,
        now: float | None = None,
    ) -> PhraseChunk:
        committed_before = self._tracker.committed_count
        tokens = prefix.tokens[:take]

        t_audio_start = self._audio_cursor
        t_audio_end = self._word_end_time(hyp, committed_before + take)

        t_wall = now if now is not None else hyp.t_wall
        chunk = PhraseChunk(
            chunk_id=self._next_id,
            text=" ".join(tokens),
            t_audio_start=t_audio_start,
            t_audio_end=t_audio_end,
            t_wall_committed=t_wall,
            reason=reason,
            word_count=take,
            stability_lag=max(0.0, t_wall - prefix.first_seen_wall),
            revised_after_commit=self._tracker.reset_contradiction(),
        )

        self._tracker.commit(take)
        self._next_id += 1
        self._audio_cursor = t_audio_end
        return chunk

    def _word_end_time(self, hyp: Hypothesis, absolute_word_index: int) -> float:
        """
        Audio position of the last word in this chunk.

        Uses the backend's own word timings when it supplies them. Otherwise
        interpolates linearly across the hypothesis span, which is fine for
        tuning but adds error to any reported p95. Backends that cannot supply
        word timings should be flagged in BackendCapabilities.
        """
        if hyp.words and absolute_word_index <= len(hyp.words):
            return hyp.words[absolute_word_index - 1].t_audio_end

        span = hyp.t_audio_end - hyp.t_audio_start
        total_words = len(hyp.text.split())
        if total_words <= 0 or span <= 0:
            return hyp.t_audio_end
        fraction = min(1.0, absolute_word_index / total_words)
        return hyp.t_audio_start + span * fraction
