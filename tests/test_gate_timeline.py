"""
The clock correction the speech gate owes everything downstream.

This is the defence for a bug that broke the only instrument this project has
for answering "is it delayed?". The gate removed silence, FastConformer
timestamped chunks against what came out of the gate, and obs.latency compared
those against a clock anchored on what went in. Every removed second was
reported as a second of latency, and it accumulated: 30.0 -> 36.7 -> 67.6 ->
89.7 -> 120.7s across five consecutive chunks on a machine at 6% CPU.

These tests need no model, so unlike tests/test_vad.py they run in CI. That is
deliberate. The gate itself is hard to test without a gigabyte of wheels; the
arithmetic that made it lie is not, and the arithmetic is what was wrong.
"""

from __future__ import annotations

import pytest

from lad_translate.adapters.vad import GateTimeline


def test_a_stream_with_no_pauses_is_unchanged():
    t = GateTimeline()
    assert t.received_time(0.0) == 0.0
    assert t.received_time(12.5) == 12.5
    assert t.suppressed_s == 0.0


def test_silence_before_the_first_word_is_charged_to_everything_after_it():
    # The speaker joins and says nothing for 30s. The first word is at speech
    # position 0 but arrived half a minute in, and a listener's latency has to
    # be measured against when it arrived.
    t = GateTimeline()
    t.hold(0.0, 30.0)
    assert t.received_time(1.0) == 31.0


def test_a_phrase_ending_where_the_pause_begins_is_not_charged_for_it():
    """
    The reason this is a breakpoint table and not a running total.

    A chunk that ends exactly when the speaker stopped arrived BEFORE the
    silence. Charging it for a pause it preceded is the original bug in
    miniature - reading the correction at emit time rather than at the
    position it applies to.
    """
    t = GateTimeline()
    t.hold(5.0, 2.0)
    assert t.received_time(5.0) == 5.0
    assert t.received_time(5.5) == 7.5


def test_pauses_accumulate():
    t = GateTimeline()
    t.hold(2.0, 1.0)
    t.hold(4.0, 3.0)
    assert t.received_time(3.0) == 4.0
    assert t.received_time(6.0) == 10.0
    assert t.suppressed_s == 4.0


def test_one_pause_is_one_breakpoint_however_many_windows_it_took():
    # feed() calls hold() once per 32ms window, so a 10s pause is 312 calls.
    # An hour of a talk must not cost an hour of tuples.
    t = GateTimeline()
    for _ in range(312):
        t.hold(4.0, 0.032)
    assert len(t._passed) == 1
    assert t.received_time(4.5) == pytest.approx(4.5 + 312 * 0.032)


def test_the_pre_roll_is_given_back():
    # 200ms was suppressed window by window, then flushed into the model when
    # the gate opened. It was received AND heard, so it is not a gap.
    t = GateTimeline()
    t.hold(3.0, 1.0)
    t.release(0.2)
    assert t.received_time(3.5) == pytest.approx(4.3)


def test_a_release_never_undercuts_an_earlier_pause():
    t = GateTimeline()
    t.hold(1.0, 5.0)
    t.hold(2.0, 0.1)
    t.release(10.0)
    # The second pause can be given back entirely; the first one still happened,
    # so 2.5s of speech is still 5.0s of silence behind the received clock.
    assert t.received_time(2.5) == pytest.approx(7.5)


def test_release_on_an_untouched_timeline_is_harmless():
    t = GateTimeline()
    t.release(0.2)
    assert t.received_time(1.0) == 1.0


def test_received_time_never_goes_backwards():
    t = GateTimeline()
    for at, held in ((0.0, 4.0), (1.5, 0.5), (1.5, 0.25), (9.0, 12.0)):
        t.hold(at, held)
    seen = [t.received_time(x / 10) for x in range(0, 200)]
    assert seen == sorted(seen)


def test_reset_forgets_the_previous_stream():
    t = GateTimeline()
    t.hold(1.0, 60.0)
    t.reset()
    assert t.received_time(1.0) == 1.0
    assert t.suppressed_s == 0.0


def test_it_removes_the_growth_that_started_this():
    """
    The live signature, reproduced and then corrected.

    A speaker who talks in short bursts with long gaps. Uncorrected, the
    reported latency of each phrase grows without bound even though every
    phrase is published the instant it is ready. Corrected, it is flat.
    """
    t = GateTimeline()
    speech_pos = 0.0
    wall = 0.0
    uncorrected: list[float] = []
    corrected: list[float] = []

    for _ in range(12):
        gap = 8.0  # the speaker pauses
        t.hold(speech_pos, gap)
        wall += gap

        speech_pos += 2.0  # then says something two seconds long
        wall += 2.0

        published_at = wall + 0.3  # 300ms to translate, synthesise and publish
        uncorrected.append(published_at - speech_pos)
        corrected.append(published_at - t.received_time(speech_pos))

    assert uncorrected[0] < uncorrected[-1] - 80, "the bug should still show"
    assert all(c == pytest.approx(0.3) for c in corrected)
