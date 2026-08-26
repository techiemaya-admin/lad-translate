"""
Drift control tests.

Written against the measured behaviour: French output ran 10.5% longer than the
English source, so a French chain accumulates roughly 0.1s of queue for every
second the speaker talks.
"""

import pytest

from lad_translate.session.drift import DriftController, DriftPolicy


def controller(**over) -> DriftController:
    return DriftController(["fr", "de"], DriftPolicy(**over))


# --- policy validation ------------------------------------------------------


def test_thresholds_must_increase():
    with pytest.raises(ValueError, match="must increase"):
        DriftPolicy(comfortable_s=2.0, speedup_at_s=1.0, skip_at_s=6.0)


def test_max_speed_below_normal_is_rejected():
    """Slowing playout would deepen the drift it is meant to relieve."""
    with pytest.raises(ValueError, match="deepen the drift"):
        DriftPolicy(max_speed=0.9)


# --- speed ------------------------------------------------------------------


def test_small_queue_runs_at_normal_speed():
    """A little queue is healthy, it absorbs jitter. Do not chase it."""
    c = controller()
    c.observe("fr", 0.4)
    assert c.speed_for("fr") == 1.0


def test_speed_stays_normal_right_up_to_the_threshold():
    c = controller(speedup_at_s=1.5)
    c.observe("fr", 1.5)
    assert c.speed_for("fr") == 1.0


def test_speed_ramps_gradually_rather_than_stepping():
    """A step change in speaking rate mid-talk is far more noticeable."""
    c = controller(speedup_at_s=1.5, skip_at_s=6.0, max_speed=1.3)
    speeds = []
    for depth in (2.0, 3.0, 4.0, 5.0):
        c.observe("fr", depth)
        speeds.append(c.speed_for("fr"))
    assert speeds == sorted(speeds), "speed must increase with queue depth"
    assert all(1.0 < s <= 1.3 for s in speeds)
    gaps = [b - a for a, b in zip(speeds, speeds[1:])]
    assert max(gaps) < 0.1, f"speed jumps too coarsely: {speeds}"


def test_speed_is_capped_at_the_ceiling():
    c = controller(max_speed=1.3, skip_at_s=6.0)
    c.observe("fr", 100.0)
    assert c.speed_for("fr") == 1.3


# --- skipping ---------------------------------------------------------------


def test_no_skipping_below_the_ceiling():
    c = controller(skip_at_s=6.0)
    c.observe("fr", 5.9)
    assert not c.should_skip("fr")


def test_skips_once_speaking_faster_is_not_enough():
    c = controller(skip_at_s=6.0)
    c.observe("fr", 6.0)
    assert c.should_skip("fr")


def test_skips_are_counted_not_silently_discarded():
    c = controller(skip_at_s=6.0)
    c.observe("fr", 7.0)
    c.note_skipped("fr", seconds=2.4, chunk_id=17)
    state = c.state("fr")
    assert state.skipped_phrases == 1
    assert state.skipped_seconds == pytest.approx(2.4)


# --- per language -----------------------------------------------------------


def test_languages_drift_independently():
    """
    French expands 10.5%, German 2.3%. One chain falling behind must not
    speed up or skip in the other.
    """
    c = controller(speedup_at_s=1.5, skip_at_s=6.0)
    c.observe("fr", 4.0)
    c.observe("de", 0.2)
    assert c.speed_for("fr") > 1.0
    assert c.speed_for("de") == 1.0
    assert not c.should_skip("de")


def test_unknown_language_is_inert_rather_than_crashing():
    """A stray chunk must not take the session down mid-event."""
    c = controller()
    c.observe("xx", 99.0)
    assert c.speed_for("xx") == 1.0
    assert not c.should_skip("xx")


# --- reporting --------------------------------------------------------------


def test_peak_depth_survives_recovery():
    """A chain that recovered still needs to show it was in trouble."""
    c = controller()
    c.observe("fr", 5.0)
    c.observe("fr", 0.1)
    assert c.state("fr").peak_depth_s == pytest.approx(5.0)
    assert c.summary()["fr"]["peak_depth_s"] == pytest.approx(5.0)


def test_negative_depth_is_clamped():
    c = controller()
    c.observe("fr", -3.0)
    assert c.state("fr").queue_depth_s == 0.0


def test_summary_covers_every_language():
    c = controller()
    assert set(c.summary()) == {"fr", "de"}


def test_realistic_expansion_triggers_speedup_before_skipping():
    """
    Simulate the measured French case: 10.5% expansion, 3 second phrases, no
    intervention. The controller should reach for speed well before it reaches
    for skipping.
    """
    c = controller(comfortable_s=0.5, speedup_at_s=1.5, skip_at_s=6.0)
    depth = 0.0
    first_speedup = None
    first_skip = None
    for phrase in range(60):
        depth += 3.0 * 0.105  # each 3s phrase adds 315ms of queue
        c.observe("fr", depth)
        if first_speedup is None and c.speed_for("fr") > 1.0:
            first_speedup = phrase
        if first_skip is None and c.should_skip("fr"):
            first_skip = phrase
            break
    assert first_speedup is not None and first_skip is not None
    assert first_speedup < first_skip, "must try speeding up before dropping speech"


# --- per-language policies --------------------------------------------------


def test_a_language_with_measurements_gets_its_own_thresholds():
    """
    Arabic peaked at 5.98s against a 6.0s skip threshold on a 45 second clip,
    where French peaked at 2.75s. One global number cannot serve both.
    """
    c = DriftController(["fr", "ar"])
    assert c.policy_for("ar").speedup_at_s < c.policy_for("fr").speedup_at_s


def test_arabic_starts_correcting_where_french_is_still_idle():
    c = DriftController(["fr", "ar"])
    c.observe("fr", 1.2)
    c.observe("ar", 1.2)
    assert c.speed_for("fr") == 1.0
    assert c.speed_for("ar") > 1.0


def test_a_language_without_measurements_gets_the_default():
    """Inventing thresholds for an unmeasured language would look like data."""
    c = DriftController(["de"], default_policy=DriftPolicy(speedup_at_s=2.0, skip_at_s=7.0))
    assert c.policy_for("de").speedup_at_s == 2.0


def test_an_explicit_policy_beats_the_measured_table():
    c = DriftController(
        ["ar"], policies={"ar": DriftPolicy(speedup_at_s=3.0, skip_at_s=9.0, max_speed=1.1)}
    )
    policy = c.policy_for("ar")
    assert policy.speedup_at_s == 3.0
    assert policy.max_speed == 1.1


def test_skipping_uses_the_per_language_threshold():
    c = DriftController(
        ["fr", "ar"],
        policies={"ar": DriftPolicy(speedup_at_s=1.0, skip_at_s=3.0)},
    )
    c.observe("fr", 3.5)
    c.observe("ar", 3.5)
    assert not c.should_skip("fr"), "French threshold is 6.0s"
    assert c.should_skip("ar"), "Arabic threshold was overridden to 3.0s"


def test_policy_for_an_unknown_language_falls_back_rather_than_raising():
    """A stray chunk must not take a session down mid-event."""
    c = DriftController(["fr"])
    assert c.policy_for("xx") is c.default_policy


def test_summary_reports_the_thresholds_that_were_in_force():
    """A post mortem needs to know what the limits were, not just the peaks."""
    c = DriftController(["fr", "ar"])
    summary = c.summary()
    assert summary["ar"]["speedup_at_s"] == c.policy_for("ar").speedup_at_s
    assert summary["fr"]["speedup_at_s"] == c.policy_for("fr").speedup_at_s
    assert summary["ar"]["speedup_at_s"] != summary["fr"]["speedup_at_s"]


def test_a_squeezed_ramp_is_rejected():
    """
    The speed ramp spans speedup_at_s to skip_at_s. Squeeze it and the voice
    jumps to max_speed rather than easing there, which is far more audible.
    """
    with pytest.raises(ValueError, match="gradual rather than a step"):
        DriftPolicy(comfortable_s=0.5, speedup_at_s=5.8, skip_at_s=6.0)


def test_measured_table_entries_are_valid_policies():
    """Guards against a typo in the table shipping a policy that cannot ramp."""
    from lad_translate.session.drift import LANGUAGE_POLICIES

    for code, policy in LANGUAGE_POLICIES.items():
        assert policy.language == code, f"{code} entry is labelled {policy.language!r}"
        assert policy.comfortable_s < policy.speedup_at_s < policy.skip_at_s
