"""
Playout drift control.

Translated speech is routinely longer than the source. Measured on the fixture:
French output ran 10.5% longer than 25.6s of English, German 2.3% longer. Each
phrase that takes longer to say than it took to hear pushes the next one
further back, and the lag compounds for as long as the speaker keeps talking.
Ten percent over a 45 minute keynote is about four and a half minutes.

session/backpressure.py does not help. That caps the queue on the way IN; drift
accumulates on the way OUT, in audio already synthesised and waiting to be
spoken. Two different queues, two different problems.

Three levers, in the order they are pulled:

    1. Speak faster. Raising the TTS rate shortens playout at some cost to
       naturalness. Cheap, reversible, and enough for normal expansion.

    2. Skip a phrase. Above the ceiling, drop one rather than fall further
       behind. The audience loses a sentence instead of sliding permanently
       out of sync.

    3. Report. Every skip is counted and logged. Silently dropping a speaker's
       words has to be visible.

There is no fourth lever. If a language chain sits at maximum speed and still
skips, the pairing of that language with that speaking rate does not work, and
that is a finding for the event, not something to tune away at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..obs.log import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class DriftPolicy:
    """
    Thresholds on the playout queue, in seconds of audio waiting to be spoken.

    The queue is the drift: audio synthesised but not yet played is exactly how
    far behind the speaker that language is.
    """

    comfortable_s: float = 0.5
    """Below this, run at normal speed. A small queue is healthy, it absorbs jitter."""

    speedup_at_s: float = 1.5
    """Start speaking faster once the queue passes this."""

    skip_at_s: float = 6.0
    """Above this, drop phrases. Speaking faster is no longer enough."""

    max_speed: float = 1.3
    """
    Ceiling on TTS rate.

    Above roughly 1.3 the output stops sounding like speech and comprehension
    falls faster than the time saved is worth. Raising this is not a free way
    to buy latency.
    """

    language: str | None = None
    """Which language this policy is for, when it came from the table below."""

    def __post_init__(self) -> None:
        if not (self.comfortable_s < self.speedup_at_s < self.skip_at_s):
            raise ValueError("thresholds must increase: comfortable < speedup < skip")
        if self.max_speed < 1.0:
            raise ValueError("max_speed below 1.0 would slow playout and deepen the drift")
        if self.skip_at_s - self.speedup_at_s < 0.5:
            # The ramp from normal speed to max_speed spans this gap. Squeeze it
            # and the voice jumps rather than eases, which is far more audible.
            raise ValueError(
                "leave at least 0.5s between speedup_at_s and skip_at_s so the "
                "speed ramp is gradual rather than a step"
            )


DEFAULT_POLICY = DriftPolicy()
"""Starting point for a language with no measurements of its own."""


LANGUAGE_POLICIES: dict[str, DriftPolicy] = {
    # MEASURED. Same 45s source, same session, nothing dropped:
    #
    #   French   peak queue 2.75s, 1 speed-up
    #   Arabic   peak queue 5.98s, 7 speed-ups
    #
    # Arabic came within 20ms of the 6.0s skip threshold on a 45 second clip.
    # Over a 45 minute keynote it would skip repeatedly. It produces more audio
    # for the same source than French does, so it needs to start correcting
    # sooner rather than harder: an earlier speedup_at_s lengthens the ramp and
    # gives the controller more total corrective capacity before the ceiling.
    #
    # max_speed is deliberately NOT raised. Whether Arabic stays intelligible
    # at 1.3x is a question for a native speaker with the actual voice, not
    # something to infer from a queue depth.
    "ar": DriftPolicy(comfortable_s=0.4, speedup_at_s=1.0, skip_at_s=6.0, language="ar"),
}
"""
Per-language overrides, and only where there are measurements.

Deliberately short. A language absent from here gets DEFAULT_POLICY, which is
honest; inventing thresholds for languages nobody has run would look like data
and would not be. Add an entry after a real session, and record the peak queue
depth and speed-up count that justified it.

Likely candidates once measured: Hindi, Urdu and Malayalam all tend to run
longer than English, and Chinese tends to run shorter.
"""


@dataclass(slots=True)
class LanguageDrift:
    language: str
    queue_depth_s: float = 0.0
    peak_depth_s: float = 0.0
    skipped_phrases: int = 0
    skipped_seconds: float = 0.0
    speedup_phrases: int = 0
    """Phrases synthesised above normal rate. A high count means this language
    expands more than the speaker leaves room for."""


class DriftController:
    """Tracks the playout queue per language and decides speed and skips."""

    def __init__(
        self,
        languages: list[str],
        default_policy: DriftPolicy | None = None,
        policies: dict[str, DriftPolicy] | None = None,
    ) -> None:
        """
        Resolution order per language: explicit `policies`, then the measured
        LANGUAGE_POLICIES table, then `default_policy`.
        """
        self.default_policy = default_policy or DEFAULT_POLICY
        overrides = policies or {}
        self._policies: dict[str, DriftPolicy] = {
            language: overrides.get(language)
            or LANGUAGE_POLICIES.get(language)
            or self.default_policy
            for language in languages
        }
        self._state: dict[str, LanguageDrift] = {
            language: LanguageDrift(language) for language in languages
        }
        for language, policy in self._policies.items():
            log.info(
                "drift policy resolved",
                extra={
                    "language": language,
                    "source": (
                        "explicit" if language in overrides
                        else "measured" if language in LANGUAGE_POLICIES
                        else "default"
                    ),
                    "speedup_at_s": policy.speedup_at_s,
                    "skip_at_s": policy.skip_at_s,
                    "max_speed": policy.max_speed,
                },
            )

    def policy_for(self, language: str) -> DriftPolicy:
        """The policy in force for a language. Falls back to the default."""
        return self._policies.get(language, self.default_policy)

    # -------------------------------------------------------------------------

    def observe(self, language: str, queue_depth_s: float) -> None:
        """Record the current playout queue depth, reported by the publisher."""
        state = self._state.get(language)
        if state is None:
            return
        state.queue_depth_s = max(0.0, queue_depth_s)
        state.peak_depth_s = max(state.peak_depth_s, state.queue_depth_s)

    def speed_for(self, language: str) -> float:
        """
        TTS rate for the next phrase in this language.

        Ramps linearly from normal at `speedup_at_s` to `max_speed` at
        `skip_at_s`, rather than switching in steps. A step change in speaking
        rate mid-talk is far more noticeable than a gradual one.
        """
        state = self._state.get(language)
        if state is None:
            return 1.0
        depth = state.queue_depth_s
        policy = self.policy_for(language)

        if depth <= policy.speedup_at_s:
            return 1.0
        span = policy.skip_at_s - policy.speedup_at_s
        fraction = min(1.0, (depth - policy.speedup_at_s) / span) if span > 0 else 1.0
        speed = 1.0 + fraction * (policy.max_speed - 1.0)
        state.speedup_phrases += 1
        return round(speed, 3)

    def should_skip(self, language: str) -> bool:
        """
        Whether to drop the next phrase in this language.

        Last resort. Speaking faster has already failed to keep up.
        """
        state = self._state.get(language)
        if state is None:
            return False
        return state.queue_depth_s >= self.policy_for(language).skip_at_s

    def note_skipped(self, language: str, seconds: float, chunk_id: int) -> None:
        state = self._state.get(language)
        if state is None:
            return
        state.skipped_phrases += 1
        state.skipped_seconds += seconds
        log.error(
            "phrase skipped to recover playout drift",
            extra={
                "language": language,
                "chunk_id": chunk_id,
                "queue_depth_s": round(state.queue_depth_s, 2),
                "skipped_phrases": state.skipped_phrases,
                "skipped_seconds": round(state.skipped_seconds, 2),
            },
        )

    # -------------------------------------------------------------------------

    def state(self, language: str) -> LanguageDrift | None:
        return self._state.get(language)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            language: {
                "queue_depth_s": round(s.queue_depth_s, 2),
                "peak_depth_s": round(s.peak_depth_s, 2),
                "skipped_phrases": s.skipped_phrases,
                "skipped_seconds": round(s.skipped_seconds, 2),
                "speedup_phrases": s.speedup_phrases,
                # Included so a post mortem can see which thresholds were in
                # force, not just how close the queue came to them.
                "speedup_at_s": self.policy_for(language).speedup_at_s,
                "skip_at_s": self.policy_for(language).skip_at_s,
            }
            for language, s in self._state.items()
        }
