"""
Session pipeline tests.

Fakes for room and adapters, so this covers the orchestration: fan-out,
per-language independence, drift response, failure containment and the session
limits. No models, no room, no network.
"""

from __future__ import annotations

import asyncio

import pytest

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
