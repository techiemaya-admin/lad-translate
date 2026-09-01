"""
Session pipeline tests.

Fakes for room and adapters, so this covers the orchestration: fan-out,
per-language independence, drift response, failure containment and the session
limits. No models, no room, no network.
"""

from __future__ import annotations

import asyncio

from lad_translate.config import SessionLimits
from lad_translate.session.drift import DriftPolicy
from lad_translate.session.pipeline import TranslationSession
from tests.fakes import FakeMt, FakeRoom, FakeStt, FakeTts, session_config

SPEECH = [
    "Good morning everyone",
    "Good morning everyone and welcome,",
    "Good morning everyone and welcome, today we begin.",
]


def build(**over):
    room = over.pop("room", None) or FakeRoom()
    stt = over.pop("stt", None) or FakeStt(SPEECH)
    mt = over.pop("mt", None) or FakeMt()
    tts = over.pop("tts", None) or FakeTts()
    config = over.pop("config", None) or session_config()
    session = TranslationSession(
        config=config, room=room, stt=stt, mt=mt, tts=tts, **over
    )
    return session, room, stt, mt, tts


# --- happy path -------------------------------------------------------------


async def test_publishes_a_track_per_language():
    session, room, *_ = build()
    await session.run()
    assert room.languages_published == ["fr", "de"]


async def test_audio_reaches_every_language():
    session, room, _, _, tts = build()
    outcome = await session.run()
    assert outcome.chunks > 0
    for language in ("fr", "de"):
        assert room.published[language], f"nothing published for {language}"
        assert tts.spoken[language]


async def test_translation_is_fanned_out_once_per_chunk():
    """One translate_many call per chunk, not one per language."""
    session, _, _, mt, _ = build()
    outcome = await session.run()
    assert len(mt.calls) == outcome.chunks


async def test_room_is_closed_even_on_the_happy_path():
    session, room, *_ = build()
    await session.run()
    assert room.closed


async def test_outcome_reports_status_and_summaries():
    session, *_ = build()
    outcome = await session.run()
    assert outcome.status == "ended"
    assert outcome.failure_reason is None
    assert set(outcome.drift) == {"fr", "de"}
    assert "frames_in" in outcome.backlog
    assert outcome.source_gaps == 0


# --- source stream continuity -----------------------------------------------
#
# t_audio counts samples that arrived. A speaker who mutes, backgrounds the page
# or rejoins produces none, so the audio clock falls behind wall time and every
# later phrase reports a latency equal to the gap. Measured on a real phone: a
# 16 minute absence made every chunk report 965s while TTS took 46ms.


async def _drive_clock(session, frames):
    """Run the frame stream through the session's clock stage."""
    return [frame async for frame in session._anchor_clock(_aiter(frames))]


async def _aiter(items):
    for item in items:
        yield item


def _frame(t_audio: float, t_wall: float):
    from lad_translate.adapters.base import AudioFrame

    return AudioFrame(b"\x00\x00" * 160, 16_000, t_audio, t_wall)


class _PhrasesStt:
    """Emits several complete phrases, so the session produces several chunks.

    FakeStt scripts one sentence being revised, which commits exactly once.
    """

    name = "fake-stt"

    def __init__(self, phrases: list[str], gap: float = 0.05) -> None:
        self._phrases = phrases
        self._gap = gap

    @property
    def required_sample_rate(self) -> int:
        return 16_000

    @property
    def emits_interims(self) -> bool:
        return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def transcribe(self, frames):
        import time

        from lad_translate.adapters.base import Hypothesis

        async for _ in frames:
            break
        wall = time.monotonic()
        for i, text in enumerate(self._phrases):
            yield Hypothesis(
                text=text,
                is_final=True,
                t_audio_start=float(i),
                t_audio_end=float(i + 1),
                t_wall=wall + i * self._gap,
                seq=i,
            )
            await asyncio.sleep(self._gap)


async def test_clock_anchors_on_the_first_frame_not_session_start():
    session, *_ = build()
    session._started_at = 0.0
    await _drive_clock(session, [_frame(0.0, 200.0), _frame(0.02, 200.02)])
    assert session.recorder.clock.wall_for(0.0) == 200.0
    assert session._source_gaps == 0


async def test_a_break_in_the_source_re_anchors_the_clock():
    session, *_ = build()
    session._started_at = 0.0
    await _drive_clock(
        session,
        [
            _frame(0.00, 100.00),
            _frame(0.02, 100.02),
            # Speaker gone for 900s: audio advanced 20ms, wall advanced 900s.
            _frame(0.04, 1000.02),
        ],
    )
    assert session._source_gaps == 1
    assert session.recorder.clock.wall_for(0.04) == 1000.02, "clock still behind the speaker"


async def test_running_late_is_not_mistaken_for_a_break():
    """
    Real latency must survive. Being behind is reported as the latency it is;
    only a genuine discontinuity is forgiven.
    """
    session, *_ = build()
    session._started_at = 0.0
    await _drive_clock(
        session,
        [
            _frame(0.0, 100.0),
            # 2s late on 1s of audio: over the SLO, well under the gap threshold.
            _frame(1.0, 103.0),
        ],
    )
    assert session._source_gaps == 0
    assert session.recorder.clock.wall_for(1.0) == 101.0, "2s of real lag was erased"


async def test_audio_arriving_in_a_burst_is_not_a_break():
    """
    When the pipeline is behind, frames queue upstream and then arrive faster
    than real time. Audio outruns wall time, which is the opposite sign, and
    must never re-anchor.
    """
    session, *_ = build()
    session._started_at = 0.0
    await _drive_clock(
        session,
        [_frame(0.0, 100.0), _frame(10.0, 100.5), _frame(20.0, 101.0)],
    )
    assert session._source_gaps == 0


# --- ordering and independence ----------------------------------------------


async def test_phrases_are_spoken_in_order_within_a_language():
    """Out of order speech is worse than late speech."""
    session, _, _, _, tts = build(stt=FakeStt(SPEECH))
    await session.run()
    spoken = tts.spoken["fr"]
    assert spoken == sorted(spoken, key=len), f"phrases out of order: {spoken}"


async def test_a_slow_language_does_not_block_a_fast_one():
    """
    The reason there is a worker per language rather than one shared queue.

    French is made slow to publish. German must still finish everything.
    """
    room = FakeRoom()
    room.push_delay["fr"] = 0.05
    session, room, _, _, tts = build(room=room)
    await session.run()
    assert len(tts.spoken["de"]) == len(tts.spoken["fr"]), (
        "the slow language held up the fast one"
    )


# --- failure containment ----------------------------------------------------


async def test_one_language_failing_does_not_silence_the_others():
    """A dead chain must not take the room down mid-keynote."""
    session, room, _, _, tts = build(tts=FakeTts(fail_for={"fr"}))
    outcome = await session.run()
    assert outcome.status == "ended"
    assert room.published["de"], "German went silent because French failed"
    assert not room.published["fr"]


async def test_empty_translation_is_skipped_not_synthesised():
    session, room, _, _, tts = build(mt=FakeMt(fail_for={"fr"}))
    await session.run()
    assert not room.published["fr"]
    assert room.published["de"]


# --- drift ------------------------------------------------------------------


async def test_normal_queue_depth_speaks_at_normal_speed():
    session, room, _, _, tts = build()
    room.depths = {"fr": 0.1, "de": 0.1}
    await session.run()
    assert all(s == 1.0 for s in tts.speeds["fr"])


async def test_growing_queue_makes_the_voice_speed_up():
    """The measured French expansion is 10.5%; this is the response to it."""
    room = FakeRoom()
    room.depths = {"fr": 4.0, "de": 0.0}
    session, room, _, _, tts = build(
        room=room, drift_policy=DriftPolicy(speedup_at_s=1.5, skip_at_s=6.0, max_speed=1.3)
    )
    await session.run()
    assert max(tts.speeds["fr"]) > 1.0
    assert all(s == 1.0 for s in tts.speeds["de"])


async def test_extreme_queue_depth_skips_the_phrase_entirely():
    room = FakeRoom()
    room.depths = {"fr": 9.0, "de": 0.0}
    session, room, _, _, tts = build(
        room=room, drift_policy=DriftPolicy(speedup_at_s=1.5, skip_at_s=6.0)
    )
    outcome = await session.run()
    assert not room.published["fr"], "should have skipped rather than fallen further behind"
    assert room.published["de"]
    assert outcome.drift["fr"]["skipped_phrases"] > 0


# --- limits -----------------------------------------------------------------


async def test_idle_cap_ends_the_session():
    """
    VOAG has no duration or idle cap at all.

    A stalled publisher must not leave language chains and a GPU running.
    """
    class SilentStt(FakeStt):
        async def transcribe(self, frames):
            await asyncio.sleep(30)
            return
            yield  # pragma: no cover

    config = session_config(limits=SessionLimits(max_idle_s=0.01, max_duration_s=3600))
    session, *_ = build(config=config, stt=SilentStt([]))
    outcome = await asyncio.wait_for(session.run(), timeout=20)
    # Reaching a limit is not a fault. A talk that finished and a publisher
    # that stalled are indistinguishable from inside the session, so calling
    # either 'failed' would make the status column meaningless.
    assert outcome.status == "ended"
    assert "no source audio" in outcome.failure_reason


async def test_duration_cap_ends_the_session():
    class EndlessStt(FakeStt):
        async def transcribe(self, frames):
            while True:
                await asyncio.sleep(30)
                yield  # pragma: no cover

    config = session_config(limits=SessionLimits(max_duration_s=0.01, max_idle_s=3600))
    session, *_ = build(config=config, stt=EndlessStt([]))
    outcome = await asyncio.wait_for(session.run(), timeout=20)
    assert outcome.status == "ended"
    assert "max_duration_s" in outcome.failure_reason


# --- storage ----------------------------------------------------------------


async def test_a_real_fault_is_still_reported_as_failed():
    """Limits end cleanly; a crash must not be dressed up as a clean end."""

    class BrokenStt(FakeStt):
        async def transcribe(self, frames):
            raise RuntimeError("STT backend died")
            yield  # pragma: no cover

    session, *_ = build(stt=BrokenStt([]))
    outcome = await session.run()
    assert outcome.status == "failed"
    assert "STT backend died" in outcome.failure_reason


async def test_a_storage_failure_does_not_kill_the_session():
    """A lost transcript row is recoverable. A dead session mid-keynote is not."""

    class BrokenStore:
        async def mark_live(self, session_id):
            return None

        async def record_transcript(self, *a, **k):
            raise RuntimeError("database gone")

        async def end_session(self, session_id, failure_reason=None):
            raise LookupError("gone")

    session, room, *_ = build(store=BrokenStore())
    outcome = await session.run()
    assert outcome.status == "ended"
    assert room.published["de"]


async def test_transcripts_store_each_chunks_own_latency():
    """
    `latency_s` sits on a per-chunk row and used to hold the session's running
    p50, so the column could never show a spike and an e2e run read back as a
    smooth decline from 8.777 to 1.688 that no chunk had actually taken.
    """

    class RecordingStore:
        def __init__(self):
            self.rows = []

        async def mark_live(self, session_id):
            return None

        async def record_transcript(self, session_id, row):
            self.rows.append(row)

        async def end_session(self, session_id, failure_reason=None):
            raise LookupError("not under test")

    from lad_translate.obs.latency import Stage

    store = RecordingStore()
    # Several separate phrases, so the chunks have genuinely different
    # latencies. The default script is one sentence revised three times and
    # commits once, and against a single sample the running p50 IS that
    # sample's value, so the two behaviours are indistinguishable.
    session, *_ = build(store=store, stt=_PhrasesStt(["First phrase.", "Second one.", "And a third."]))

    # Record what the recorder reported for each chunk as it closed, which is
    # the only moment the per-chunk figure exists.
    measured: dict[tuple[int, str], float | None] = {}
    real_mark = session.recorder.mark

    def spy(chunk_id, language, stage, t_wall):
        result = real_mark(chunk_id, language, stage, t_wall)
        if stage is Stage.PUBLISHED:
            measured[(chunk_id, language)] = result
        return result

    session.recorder.mark = spy
    await session.run()

    stored = {(r.chunk_id, r.language): r.latency_s for r in store.rows}
    assert stored, "no transcripts were written"
    assert measured, "no chunk was published"
    for key, value in measured.items():
        assert key in stored, f"chunk {key} published but never persisted"
        assert stored[key] == value, (
            f"chunk {key} stored {stored[key]} but actually took {value}"
        )


async def test_session_applies_the_measured_table_per_language():
    """
    Arabic builds queue far faster than French for the same source: measured
    peaks were 5.98s against 2.75s. At a queue depth between the two
    thresholds, Arabic should be correcting while French is not.
    """
    room = FakeRoom()
    room.depths = {"fr": 1.2, "ar": 1.2}
    session, room, _, _, tts = build(
        room=room, config=session_config(targets=("fr", "ar"))
    )
    await session.run()
    assert all(s == 1.0 for s in tts.speeds["fr"]), "French threshold is 1.5s"
    assert max(tts.speeds["ar"]) > 1.0, "Arabic threshold is 1.0s"


async def test_an_explicit_policy_overrides_the_table_through_the_session():
    """
    Uses two languages that both take the default, so the only thing that can
    make them differ is the explicit override. Overriding Arabic would prove
    nothing: the measured table already gives it a lower threshold.
    """
    room = FakeRoom()
    room.depths = {"fr": 1.2, "de": 1.2}
    session, room, _, _, tts = build(
        room=room,
        config=session_config(targets=("fr", "de")),
        drift_policies={"de": DriftPolicy(speedup_at_s=0.8, skip_at_s=6.0)},
    )
    await session.run()
    assert all(s == 1.0 for s in tts.speeds["fr"]), "French kept the default 1.5s"
    assert max(tts.speeds["de"]) > 1.0, "German was overridden to 0.8s"
