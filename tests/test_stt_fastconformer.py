"""
FastConformer: the parts that can be checked without a GPU.

NeMo needs torch with CUDA and neither is installed here, so the tensor path
is unverified until it runs on the A4000 and this suite must not pretend
otherwise. What IS tested is everything that decides WHICH audio reaches the
encoder -- the frame arithmetic, the buffer bounds, the word stamping -- which
is where a streaming adapter actually goes wrong. An off-by-one in the
pre-encode cache does not raise; it quietly degrades the transcript, and the
only way to catch that is here, before the hardware arrives.
"""

from __future__ import annotations

import pytest

from lad_translate.adapters.registry import STT_BACKENDS, build_stt
from lad_translate.adapters.stt_fastconformer import (
    DEFAULT_LOOKAHEAD,
    LOOKAHEADS,
    ChunkSchedule,
    FrameWindow,
    StreamingGeometry,
    WordClock,
    _extract_text,
    iter_lookaheads,
    lookahead_ms,
)

# The published geometry for the 480ms lookahead, as NeMo's setup_streaming_params
# computes it: sampling_frames 8, subsampling_factor 8, att_context_size [70, 6].
#   chunk = 8 + 8*6 = 56 frames = 560ms
#   shift = 8 + 8*6 = 56 frames  (cache_drop_size is 0 for chunked_limited)
GEOMETRY = StreamingGeometry(
    chunk_size_first=56,
    chunk_size=56,
    shift_size_first=56,
    shift_size=56,
    pre_encode_cache_first=0,
    pre_encode_cache=8,
    drop_extra_pre_encoded=1,
)


# --- the lookahead table ----------------------------------------------------


@pytest.mark.parametrize("name,spec", sorted(LOOKAHEADS.items()))
def test_each_lookahead_matches_the_formula_it_claims(name, spec):
    """
    ms = att_context_size[1] * subsampling_factor * window_stride.

    The table is quoted in the module docstring and in the registry note, and
    a wrong number there would be repeated into a latency budget.
    """
    assert lookahead_ms(spec.att_context_size) == spec.ms
    assert name == f"{spec.ms}ms"


def test_more_lookahead_buys_lower_wer():
    """Sanity on the transcribed table: the trade has to run in one direction."""
    ordered = [spec for _, spec in iter_lookaheads()]
    assert [s.ms for s in ordered] == sorted(s.ms for s in ordered)
    assert [s.wer_rnnt for s in ordered] == sorted((s.wer_rnnt for s in ordered), reverse=True)
    assert [s.wer_ctc for s in ordered] == sorted((s.wer_ctc for s in ordered), reverse=True)


def test_rnnt_beats_ctc_at_every_lookahead():
    """If this ever inverts, the default decoder choice needs revisiting."""
    assert all(s.wer_rnnt < s.wer_ctc for s in LOOKAHEADS.values())


def test_the_default_is_not_nemos_default():
    """
    NeMo ships 1040ms. This project cannot spend half a 2s budget before the
    first word reaches the translator, and 480ms buys 560ms back for 0.3 WER
    points. If someone "fixes" this to match upstream, that is a regression.
    """
    assert DEFAULT_LOOKAHEAD == "480ms"
    assert LOOKAHEADS[DEFAULT_LOOKAHEAD].ms == 480
    assert LOOKAHEADS["1040ms"].ms - LOOKAHEADS[DEFAULT_LOOKAHEAD].ms == 560


# --- geometry normalisation -------------------------------------------------


def test_int_and_list_streaming_cfg_values_normalise_the_same_way():
    """
    NeMo stores these as either an int or a [first_step, steady_state] pair,
    and branches on that in four separate places. Doing it once is the whole
    reason StreamingGeometry exists.
    """
    assert StreamingGeometry._pair(56) == (56, 56)
    assert StreamingGeometry._pair([48, 56]) == (48, 56)
    assert StreamingGeometry._pair((48, 56)) == (48, 56)


def test_an_unexpected_streaming_cfg_shape_is_refused_not_guessed():
    with pytest.raises(ValueError, match="2 element"):
        StreamingGeometry._pair([1, 2, 3])


def test_frames_convert_to_audio_seconds_at_10ms_a_frame():
    assert GEOMETRY.audio_time(0) == 0.0
    assert GEOMETRY.audio_time(100) == pytest.approx(1.0)
    assert GEOMETRY.step_interval_s == pytest.approx(0.56)


# --- the schedule -----------------------------------------------------------


def test_a_partial_chunk_yields_nothing():
    """
    The encoder is entitled to a full chunk. Handing it a short one produces
    output for audio that has not arrived, which is exactly the class of bug
    that made Whisper invent speech out of silence.
    """
    schedule = ChunkSchedule(GEOMETRY)
    assert schedule.offer(55) == []
    assert schedule.offer(0) == []


def test_the_first_step_drops_nothing_because_nothing_was_prepended():
    """
    Mirrors NeMo's calc_drop_extra_pre_encoded, which special-cases step 0 for
    this reason: no caching has happened yet, so there is no cache output to
    discard. Dropping anyway would eat the first word of the session.
    """
    plan = ChunkSchedule(GEOMETRY).offer(56)[0]
    assert plan.step == 0
    assert plan.drop_extra_pre_encoded == 0
    assert plan.cache_frames == 0
    assert plan.zero_pad == 0
    assert (plan.start, plan.end) == (0, 56)


def test_later_steps_carry_real_history_and_drop_its_output():
    schedule = ChunkSchedule(GEOMETRY)
    schedule.offer(56)
    second = schedule.offer(112)[0]
    assert second.step == 1
    assert second.drop_extra_pre_encoded == GEOMETRY.drop_extra_pre_encoded
    assert second.cache_frames == GEOMETRY.pre_encode_cache
    assert second.zero_pad == 0
    assert (second.cache_start, second.start, second.end) == (48, 56, 112)


def test_the_cache_is_always_filled_to_size_by_padding_when_history_is_short():
    """Near the start there is not enough history, and the shortfall is zeros."""
    geometry = StreamingGeometry(
        chunk_size_first=10,
        chunk_size=10,
        shift_size_first=4,
        shift_size=4,
        pre_encode_cache_first=0,
        pre_encode_cache=8,
        drop_extra_pre_encoded=1,
    )
    schedule = ChunkSchedule(geometry)
    plans = schedule.offer(30)
    assert [p.start for p in plans] == [0, 4, 8, 12, 16, 20]
    for plan in plans[1:]:
        expected = geometry.pre_encode_cache
        assert plan.cache_frames + plan.zero_pad == expected
    assert plans[1].zero_pad == 4, "only 4 frames of history existed at frame 4"
    assert plans[2].zero_pad == 0, "by frame 8 the cache is full"


def test_chunk_width_accounts_for_everything_handed_to_the_encoder():
    schedule = ChunkSchedule(GEOMETRY)
    schedule.offer(56)
    plan = schedule.offer(112)[0]
    assert plan.width == plan.zero_pad + plan.cache_frames + (plan.end - plan.start)
    assert plan.width == 8 + 56


def test_audio_arriving_in_one_burst_produces_the_same_steps_as_a_trickle():
    """
    The room does not deliver frames on chunk boundaries, so the schedule has
    to be indifferent to how audio is grouped on the way in.
    """
    burst = ChunkSchedule(GEOMETRY).offer(560)
    trickle = []
    schedule = ChunkSchedule(GEOMETRY)
    for available in range(1, 561):
        trickle.extend(schedule.offer(available))
    assert burst == trickle
    assert len(burst) == 10


# --- the bound that makes a long talk possible ------------------------------


def test_memory_held_stays_flat_across_a_ninety_minute_session():
    """
    THE point of reimplementing NeMo's buffer. Theirs pads and rewrites the
    whole tensor on every append and never releases consumed audio, which is
    O(n^2) work and unbounded memory over a keynote. Here what is retained is
    one chunk plus one pre-encode cache, whatever the hour.
    """
    schedule = ChunkSchedule(GEOMETRY)
    window = FrameWindow()
    ninety_minutes = 90 * 60 * 100  # frames, at 10ms each
    held = []

    for _ in range(ninety_minutes // 56):
        window.extend(56)
        for _ in schedule.offer(window.total):
            pass
        window.drop_before(schedule.retain_from)
        held.append(window.held)

    assert max(held) == min(held[2:]), "retained frames must not grow with session length"
    assert max(held) <= GEOMETRY.chunk_size + GEOMETRY.pre_encode_cache
    assert window.total == (ninety_minutes // 56) * 56, "the absolute clock still counts everything"


def test_retain_from_never_discards_history_the_next_step_needs():
    """
    The schedule and the buffer agree or the transcript is quietly wrong. Run
    them against each other: every step must be sliceable from a window that
    has been trimmed to whatever the schedule said was safe to drop.
    """
    schedule = ChunkSchedule(GEOMETRY)
    window = FrameWindow()
    steps = 0

    for _ in range(40):
        window.extend(56)
        for plan in schedule.offer(window.total):
            # Raises if retain_from ever threw away audio this step still needs.
            window.local(plan.cache_start, plan.end)
            steps += 1
        window.drop_before(schedule.retain_from)

    assert steps == 40, "one step per chunk, none skipped and none replayed"


def test_a_burst_of_audio_yields_several_steps_that_all_keep_their_history():
    """
    Audio does not arrive one chunk at a time. After a reconnect, or behind a
    slow first inference, several chunks become runnable at once, and offer()
    hands back the whole batch with the cursor already advanced past all of
    them. Retention is therefore a per-BATCH operation -- see transcribe().
    """
    schedule = ChunkSchedule(GEOMETRY)
    window = FrameWindow()
    window.extend(56 * 5)

    plans = schedule.offer(window.total)
    assert len(plans) == 5, "the burst must actually produce a batch, or this proves nothing"
    for plan in plans:
        window.local(plan.cache_start, plan.end)
    window.drop_before(schedule.retain_from)

    assert window.held <= GEOMETRY.chunk_size + GEOMETRY.pre_encode_cache


def test_discarding_between_steps_of_one_batch_is_exactly_what_the_window_catches():
    """
    The hazard the comment in transcribe() is guarding against, pinned here so
    that moving the discard back inside the loop fails loudly rather than
    producing a subtly wrong transcript.
    """
    schedule = ChunkSchedule(GEOMETRY)
    window = FrameWindow()
    window.extend(56 * 5)
    plans = schedule.offer(window.total)

    with pytest.raises(IndexError, match="already discarded"):
        for plan in plans:
            window.local(plan.cache_start, plan.end)
            window.drop_before(schedule.retain_from)  # wrong: batch is not finished


# --- the flush --------------------------------------------------------------


def test_the_tail_of_the_last_sentence_is_not_dropped():
    """
    Mid-session a short chunk means "wait". At end of session it means "this
    is all there is", and discarding it loses the speaker's closing words.
    """
    schedule = ChunkSchedule(GEOMETRY)
    schedule.offer(56)
    tail = schedule.flush(80)
    assert tail is not None
    assert (tail.start, tail.end) == (56, 80)


def test_flushing_twice_does_not_replay_audio():
    schedule = ChunkSchedule(GEOMETRY)
    schedule.offer(56)
    assert schedule.flush(80) is not None
    assert schedule.flush(80) is None


def test_flush_with_nothing_pending_is_a_no_op():
    schedule = ChunkSchedule(GEOMETRY)
    schedule.offer(56)
    assert schedule.flush(56) is None


# --- the window -------------------------------------------------------------


def test_slicing_discarded_audio_raises_rather_than_returning_the_wrong_frames():
    """
    Silent misalignment here would show up as a transcript that is subtly
    wrong and nothing else. Loud is better.
    """
    window = FrameWindow()
    window.extend(100)
    window.drop_before(40)
    assert window.local(40, 60) == (0, 20)
    with pytest.raises(IndexError, match="already discarded"):
        window.local(39, 60)


def test_slicing_audio_that_has_not_arrived_raises():
    window = FrameWindow()
    window.extend(50)
    with pytest.raises(IndexError, match="has not arrived"):
        window.local(0, 51)


def test_dropping_is_clamped_and_idempotent():
    window = FrameWindow()
    window.extend(100)
    assert window.drop_before(30) == 30
    assert window.drop_before(30) == 0, "dropping the same point twice drops nothing"
    assert window.drop_before(10) == 0, "dropping backwards is refused, not honoured"
    assert window.drop_before(500) == 70, "cannot drop past what has arrived"
    assert window.held == 0


# --- word stamping ----------------------------------------------------------


def test_words_keep_the_audio_position_of_the_step_that_first_produced_them():
    clock = WordClock()
    clock.stamp("we are", 1.0)
    timings = clock.stamp("we are going to Dubai", 2.0)
    assert [w.word for w in timings] == ["we", "are", "going", "to", "Dubai"]
    assert [w.t_audio_end for w in timings] == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_word_spans_are_contiguous_and_monotonic():
    clock = WordClock()
    clock.stamp("one two", 1.0)
    timings = clock.stamp("one two three", 2.0)
    assert timings[0].t_audio_start == 0.0
    for earlier, later in zip(timings, timings[1:]):
        assert earlier.t_audio_end == later.t_audio_start
        assert later.t_audio_end >= later.t_audio_start


def test_a_revised_word_is_restamped_and_so_is_everything_after_it():
    """
    RNNT never does this, but CTC re-decodes its whole prefix every step. A
    timing left attached to text that has since changed is a latency figure
    describing audio that was never spoken.
    """
    clock = WordClock()
    clock.stamp("we are going to do buy", 1.0)
    timings = clock.stamp("we are going to Dubai", 2.0)
    assert [w.word for w in timings] == ["we", "are", "going", "to", "Dubai"]
    assert [w.t_audio_end for w in timings] == [1.0, 1.0, 1.0, 1.0, 2.0]


def test_an_unchanged_hypothesis_does_not_move_any_stamp():
    clock = WordClock()
    first = clock.stamp("steady as she goes", 1.0)
    second = clock.stamp("steady as she goes", 9.0)
    assert first == second


def test_the_stamp_is_a_ceiling_not_an_estimate():
    """
    Every word is dated to the END of the step that produced it, so a latency
    computed from it can only over-report. On a project whose claim is that
    its numbers are measured rather than estimated, erring optimistic would be
    the worse failure.
    """
    clock = WordClock()
    timings = clock.stamp("hello there", 0.56)
    assert all(w.t_audio_end <= 0.56 for w in timings)
    assert timings[-1].t_audio_end == 0.56


# --- decoder output shapes --------------------------------------------------


def test_ctc_and_rnnt_transcripts_are_read_from_the_same_return_slot():
    """
    conformer_stream_step returns a list of strings for CTC and a list of
    Hypothesis objects for RNNT, through the same position in the tuple.
    """

    class FakeHypothesis:
        text = "  from an rnnt hypothesis  "

    assert _extract_text(["  from ctc  "]) == "from ctc"
    assert _extract_text([FakeHypothesis()]) == "from an rnnt hypothesis"


def test_an_empty_step_reads_as_empty_rather_than_raising():
    """Steps that land entirely inside a pause decode to nothing. Routine."""
    assert _extract_text([]) == ""
    assert _extract_text(None) == ""
    assert _extract_text([""]) == ""


# --- construction and registry ----------------------------------------------


def test_a_missing_dependency_is_reported_at_construction():
    """Not when the audience is already in the room."""
    with pytest.raises(RuntimeError, match="nemo_toolkit"):
        build_stt("fastconformer")


def test_the_registry_records_that_this_model_is_english_only():
    """
    "multi" in stt_en_fastconformer_hybrid_large_streaming_multi means multiple
    lookaheads, not multilingual. Misreading it would put an Arabic speaker in
    front of an English-only recogniser.
    """
    note = STT_BACKENDS["fastconformer"].note.lower()
    assert "english only" in note
    assert "not " in note and "multilingual" in note


def test_the_registry_records_that_it_is_unrun_and_gpu_only():
    spec = STT_BACKENDS["fastconformer"]
    assert spec.credible_on == frozenset({"cuda"})
    assert "never run" in spec.note.lower()


def test_whisper_is_still_credible_nowhere():
    """
    The reason this adapter exists. Whisper's window is architectural, so
    adding a better option does not rehabilitate it.
    """
    assert STT_BACKENDS["faster-whisper"].credible_on == frozenset()
