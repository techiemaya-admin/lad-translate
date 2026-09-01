import pytest

from lad_translate.obs.latency import AudioClock, LatencyRecorder, Stage


def test_audio_clock_maps_audio_position_to_wall_time():
    clock = AudioClock()
    clock.anchor(t_audio=0.0, t_wall=500.0)
    assert clock.wall_for(3.0) == 503.0


def test_audio_clock_anchor_is_set_once():
    clock = AudioClock()
    clock.anchor(0.0, 500.0)
    clock.anchor(10.0, 999.0)
    assert clock.wall_for(0.0) == 500.0, "re-anchoring would hide publisher clock drift"


def test_unanchored_clock_refuses_to_guess():
    with pytest.raises(RuntimeError):
        AudioClock().wall_for(1.0)


def test_rebase_moves_the_epoch_where_anchor_will_not():
    clock = AudioClock()
    clock.anchor(t_audio=0.0, t_wall=500.0)
    # The speaker went away for 900s: audio advanced 2s, wall advanced 902s.
    clock.rebase(t_audio=2.0, t_wall=1402.0)
    assert clock.wall_for(2.0) == 1402.0


def test_rebase_reports_how_far_the_clock_had_fallen_behind():
    clock = AudioClock()
    clock.anchor(t_audio=0.0, t_wall=500.0)
    assert clock.rebase(t_audio=2.0, t_wall=1402.0) == pytest.approx(900.0)


def test_rebase_on_a_fresh_clock_reports_no_correction():
    """Nothing was wrong yet, so there is no gap to attribute to the publisher."""
    assert AudioClock().rebase(t_audio=0.0, t_wall=500.0) == 0.0


def test_latency_after_a_gap_is_the_real_one_not_the_gap():
    """
    The bug this exists for: a phone that left and rejoined 16 minutes later
    made every subsequent chunk report ~965s, while translate and TTS were
    measured in tens of milliseconds.
    """
    rec = LatencyRecorder(slo_seconds=2.0)
    rec.clock.anchor(t_audio=0.0, t_wall=1000.0)

    # Speaker vanishes; audio reaches 2.0s only 900s later.
    rec.clock.rebase(t_audio=2.0, t_wall=1900.0)
    rec.open_chunk(0, "fr", t_audio_end=3.0)
    rec.mark(0, "fr", Stage.COMMITTED, 1901.2)
    rec.mark(0, "fr", Stage.TRANSLATED, 1901.3)
    rec.mark(0, "fr", Stage.TTS_FIRST_AUDIO, 1901.5)
    rec.mark(0, "fr", Stage.PUBLISHED, 1901.6)

    stats = rec.stats("fr")
    assert stats.p50 == pytest.approx(0.6)
    assert not stats.breached(2.0), "the gap must not be charged to the pipeline"


def test_publish_mark_returns_this_chunks_latency():
    """
    The per-chunk figure is only available at the moment the trace closes, and
    the transcript row needs exactly that rather than the session's running p50.
    """
    rec = _recorder()
    rec.open_chunk(0, "fr", t_audio_end=10.0)
    assert rec.mark(0, "fr", Stage.COMMITTED, 1010.4) is None
    assert rec.mark(0, "fr", Stage.PUBLISHED, 1010.9) == pytest.approx(0.9)


def test_publish_mark_returns_each_chunks_own_latency_not_an_average():
    rec = _recorder()
    rec.open_chunk(0, "fr", t_audio_end=10.0)
    rec.mark(0, "fr", Stage.PUBLISHED, 1014.0)
    rec.open_chunk(1, "fr", t_audio_end=20.0)
    second = rec.mark(1, "fr", Stage.PUBLISHED, 1020.5)

    assert second == pytest.approx(0.5), "a spike-free series means the median leaked in"
    assert rec.stats("fr").p50 == pytest.approx(2.25)


def test_mark_for_an_unknown_chunk_returns_none():
    assert _recorder().mark(99, "fr", Stage.PUBLISHED, 1000.0) is None


def _recorder() -> LatencyRecorder:
    rec = LatencyRecorder(slo_seconds=2.0)
    rec.clock.anchor(t_audio=0.0, t_wall=1000.0)
    return rec


def test_glass_to_glass_is_measured_from_end_of_source_phrase():
    rec = _recorder()
    rec.open_chunk(chunk_id=0, language="ar", t_audio_end=10.0)
    # Source phrase ended at audio 10.0, so wall 1010.0.
    rec.mark(0, "ar", Stage.COMMITTED, 1010.4)
    rec.mark(0, "ar", Stage.TRANSLATED, 1010.5)
    rec.mark(0, "ar", Stage.TTS_FIRST_AUDIO, 1010.7)
    rec.mark(0, "ar", Stage.PUBLISHED, 1010.9)

    stats = rec.stats("ar")
    assert stats.count == 1
    assert stats.p50 == pytest.approx(0.9)
    assert not stats.breached(2.0)


def test_stage_breakdown_accounts_for_the_whole_budget():
    rec = _recorder()
    rec.open_chunk(0, "fr", t_audio_end=5.0)
    rec.mark(0, "fr", Stage.COMMITTED, 1005.4)
    rec.mark(0, "fr", Stage.TRANSLATED, 1005.6)
    rec.mark(0, "fr", Stage.TTS_FIRST_AUDIO, 1005.9)
    rec.mark(0, "fr", Stage.PUBLISHED, 1006.1)

    stages = rec.stats("fr").stage_means
    assert stages["chunker"] == pytest.approx(0.4)
    assert stages["translate"] == pytest.approx(0.2)
    assert stages["tts"] == pytest.approx(0.3)
    assert stages["publish"] == pytest.approx(0.2)
    assert sum(stages.values()) == pytest.approx(1.1), "stages must sum to glass to glass"


def test_slo_breach_is_counted_per_chunk():
    rec = _recorder()
    for i, published in enumerate((1000.5, 1003.5)):
        rec.open_chunk(i, "es", t_audio_end=0.0)
        rec.mark(i, "es", Stage.PUBLISHED, published)
    summary = rec.summary()
    assert summary["slo_breaches"] == 1
    assert summary["languages"]["es"]["breached"] is True


def test_languages_are_tracked_independently():
    rec = _recorder()
    for lang, published in (("ar", 1000.8), ("de", 1002.9)):
        rec.open_chunk(0, lang, t_audio_end=0.0)
        rec.mark(0, lang, Stage.PUBLISHED, published)
    assert rec.stats("ar").p95 == pytest.approx(0.8)
    assert rec.stats("de").p95 == pytest.approx(2.9)


def test_unpublished_chunks_are_reported_not_silently_dropped():
    rec = _recorder()
    rec.open_chunk(0, "ar", t_audio_end=0.0)
    rec.mark(0, "ar", Stage.COMMITTED, 1000.3)
    assert rec.summary()["unfinished_chunks"] == 1


def test_repeat_mark_keeps_the_first_because_that_is_what_was_heard():
    rec = _recorder()
    trace = rec.open_chunk(0, "ar", t_audio_end=0.0)
    rec.mark(0, "ar", Stage.PUBLISHED, 1000.5)
    trace.mark(Stage.PUBLISHED, 1009.0)
    assert trace.glass_to_glass == pytest.approx(0.5)
