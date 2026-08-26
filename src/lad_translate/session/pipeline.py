"""
Translation session orchestrator.

    source track -> backlog guard -> STT -> chunker
                 -> translate fan-out -> per-language worker -> language track

One worker task per language, each with its own ordered queue. Phrases within a
language must be spoken in the order they were said, but a language that falls
behind must not hold up the others. A single shared queue would give ordering
and coupling; a task per chunk would give independence and scrambled speech.
A worker per language gives both.

Translation fans out concurrently inside the MT adapter, so the five language
chains diverge only at synthesis, which is where their costs actually differ.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass

from ..adapters.base import MtAdapter, SttAdapter, TtsAdapter, VoiceSpec
from ..chunker import PhraseChunk, PhraseChunker
from ..config import SessionConfig
from ..db.sessions import SessionStore, TranscriptRow
from ..obs.latency import LatencyRecorder, Stage
from ..obs.log import get_logger
from .backpressure import BacklogGuard, guarded
from .drift import DriftController, DriftPolicy
from .room import TranslationRoom

log = get_logger(__name__)


@dataclass(slots=True)
class SessionOutcome:
    session_id: str
    status: str
    chunks: int
    failure_reason: str | None
    latency: dict
    backlog: dict
    drift: dict


class _LanguageWorker:
    """Synthesises and publishes one language, in order, at its own pace."""

    def __init__(
        self,
        language: str,
        session: TranslationSession,
    ) -> None:
        self.language = language
        self._session = session
        self._queue: asyncio.Queue[tuple[PhraseChunk, str] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"lang-{self.language}")

    def submit(self, chunk: PhraseChunk, text: str) -> None:
        self._queue.put_nowait((chunk, text))

    async def stop(self) -> None:
        self._queue.put_nowait(None)
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            chunk, text = item
            try:
                await self._speak(chunk, text)
            except Exception:
                # One bad phrase must not silence the language for the rest of
                # the event. Log it, drop it, carry on.
                log.exception(
                    "phrase failed",
                    extra={"language": self.language, "chunk_id": chunk.chunk_id},
                )

    async def _speak(self, chunk: PhraseChunk, text: str) -> None:
        session = self._session
        if not text.strip():
            return

        # Read the playout queue before committing to synthesis, so the drift
        # decision is made on current state rather than on state from before
        # the previous phrase was spoken.
        session.drift.observe(self.language, session.room.queue_depth(self.language))

        if session.drift.should_skip(self.language):
            session.drift.note_skipped(self.language, chunk.t_audio_end - chunk.t_audio_start, chunk.chunk_id)
            await session.persist(chunk, self.language, text, latency=None)
            return

        voice = VoiceSpec(
            language=self.language,
            voice_id=session.config.voice_for(self.language),
            speed=session.drift.speed_for(self.language),
        )

        first = True
        async for speech in session.tts.synthesise(text, voice, chunk.chunk_id):
            if first:
                session.recorder.mark(
                    chunk.chunk_id, self.language, Stage.TTS_FIRST_AUDIO, speech.t_wall
                )
            await session.room.push(self.language, speech.pcm, speech.sample_rate)
            if first:
                session.recorder.mark(
                    chunk.chunk_id, self.language, Stage.PUBLISHED, time.monotonic()
                )
                first = False

        stats = session.recorder.stats(self.language)
        await session.persist(chunk, self.language, text, latency=stats.p50 if stats.count else None)


class TranslationSession:
    """One event: connect, translate, publish, settle."""

    def __init__(
        self,
        config: SessionConfig,
        room: TranslationRoom,
        stt: SttAdapter,
        mt: MtAdapter,
        tts: TtsAdapter,
        store: SessionStore | None = None,
        drift_policy: DriftPolicy | None = None,
        drift_policies: dict[str, DriftPolicy] | None = None,
        max_lag_s: float = 3.0,
    ) -> None:
        self.config = config
        self.room = room
        self.stt = stt
        self.mt = mt
        self.tts = tts
        self.store = store

        self.chunker = PhraseChunker(config.chunker)
        self.recorder = LatencyRecorder(slo_seconds=config.slo_seconds)
        self.guard = BacklogGuard(max_lag_s=max_lag_s)
        # drift_policy is the fallback; per-language thresholds come from
        # drift_policies or the measured table in session/drift.py. Arabic
        # builds queue far faster than French for the same source, so one
        # global threshold is wrong for both.
        self.drift = DriftController(
            config.target_codes, default_policy=drift_policy, policies=drift_policies
        )

        self._workers: dict[str, _LanguageWorker] = {}
        self._chunks = 0
        self._started_at = 0.0
        self._last_audio_at = 0.0
        self._stop = asyncio.Event()
        self._failure: str | None = None
        self._end_reason: str | None = None
        """
        Why the session stopped, when it stopped normally.

        Reaching a limit is not a failure. A talk that finished and a publisher
        that stalled are indistinguishable from in here, and calling either one
        'failed' means the status column stops telling anyone anything. Faults
        are crashes and lost rooms; those set _failure.
        """

    # -------------------------------------------------------------------------

    async def run(self) -> SessionOutcome:
        self._started_at = time.monotonic()
        self._last_audio_at = self._started_at
        log_ = log.bind(session_id=self.config.session_id, tenant_id=self.config.tenant.tenant_id)

        await self.room.publish_languages(self.config.target_codes)
        for language in self.config.target_codes:
            worker = _LanguageWorker(language, self)
            worker.start()
            self._workers[language] = worker

        if self.store is not None:
            await self.store.mark_live(self.config.session_id)

        watchdog = asyncio.create_task(self._watchdog(), name="limits")
        pump = asyncio.create_task(self._pump(), name="pump")
        stop = asyncio.create_task(self._stop.wait(), name="stop")
        try:
            # The limits have to be able to interrupt a pump that is blocked,
            # not just ask it to notice. A hung STT backend never returns from
            # transcribe(), so a cooperative flag would leave the session
            # running for ever and the caps would protect nothing. Race the
            # pump against the stop signal and cancel whichever loses.
            done, pending = await asyncio.wait(
                {pump, stop}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if pump in done:
                # Re-raise whatever the pump failed with.
                pump.result()
        except Exception as exc:
            self._failure = self._failure or f"{type(exc).__name__}: {exc}"
            log_.exception("session failed")
        finally:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog
            await self._shutdown()

        return await self._settle()

    # -------------------------------------------------------------------------

    async def _pump(self) -> None:
        """Drive audio through STT and the chunker until the source ends."""
        frames = guarded(self.room.source_frames(), self.guard)

        async for hypothesis in self.stt.transcribe(frames):
            if self._stop.is_set():
                break
            self._last_audio_at = time.monotonic()
            if not self.recorder.clock.anchored:
                # Anchor from the first hypothesis, whose t_audio_end maps to
                # now minus however long the backend took. Anchoring on the
                # first audio frame instead would be more accurate; this is
                # what is available without threading the clock through STT.
                self.recorder.clock.anchor(t_audio=0.0, t_wall=self._started_at)

            for chunk in self.chunker.feed(hypothesis):
                await self._dispatch(chunk)

            for chunk in self.chunker.tick(time.monotonic()):
                await self._dispatch(chunk)

        for chunk in self.chunker.flush():
            await self._dispatch(chunk)

    async def _dispatch(self, chunk: PhraseChunk) -> None:
        """Translate one phrase and hand it to every language worker."""
        self._chunks += 1
        targets = self.config.target_codes

        for language in targets:
            self.recorder.open_chunk(chunk.chunk_id, language, chunk.t_audio_end)
            self.recorder.mark(chunk.chunk_id, language, Stage.COMMITTED, chunk.t_wall_committed)

        if chunk.revised_after_commit:
            self.recorder.record_revision(chunk.chunk_id)

        translations = await self.mt.translate_many(
            chunk.text, self.config.source_language, targets
        )
        now = time.monotonic()
        for language, text in translations.items():
            self.recorder.mark(chunk.chunk_id, language, Stage.TRANSLATED, now)
            self._workers[language].submit(chunk, text)

    # -------------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """
        Enforce the session limits.

        VOAG has no duration or cost cap at all, and event sessions run long. A
        session that nobody ends keeps five language chains and a GPU busy
        indefinitely.
        """
        limits = self.config.limits
        warned_duration = False
        while not self._stop.is_set():
            await asyncio.sleep(5.0)
            now = time.monotonic()
            elapsed = now - self._started_at
            idle = now - self._last_audio_at

            if elapsed >= limits.max_duration_s:
                self._end_reason = f"reached max_duration_s ({limits.max_duration_s}s)"
                log.warning("session duration cap reached", extra={"elapsed_s": round(elapsed)})
                self._stop.set()
                return
            if idle >= limits.max_idle_s:
                self._end_reason = f"no source audio for {limits.max_idle_s}s"
                log.info("session idle cap reached", extra={"idle_s": round(idle)})
                self._stop.set()
                return
            if not warned_duration and elapsed >= limits.max_duration_s * limits.warn_at_fraction:
                warned_duration = True
                log.warning(
                    "session approaching duration cap",
                    extra={"elapsed_s": round(elapsed), "cap_s": limits.max_duration_s},
                )

    # -------------------------------------------------------------------------

    async def persist(
        self, chunk: PhraseChunk, language: str, text: str, latency: float | None
    ) -> None:
        if self.store is None:
            return
        try:
            await self.store.record_transcript(
                self.config.session_id,
                TranscriptRow(
                    chunk_id=chunk.chunk_id,
                    language=language,
                    source_text=chunk.text,
                    translated_text=text,
                    t_audio_start=chunk.t_audio_start,
                    t_audio_end=chunk.t_audio_end,
                    latency_s=latency,
                    commit_reason=chunk.reason.value,
                    revised=chunk.revised_after_commit,
                ),
            )
        except Exception:
            # Storage must never take the room down. A lost transcript row is
            # recoverable; a dead session mid-keynote is not.
            log.exception(
                "transcript write failed",
                extra={"chunk_id": chunk.chunk_id, "language": language},
            )

    async def _shutdown(self) -> None:
        for worker in self._workers.values():
            await worker.stop()
        await self.room.close()

    async def _settle(self) -> SessionOutcome:
        latency = self.recorder.summary()
        backlog = self.guard.summary()
        drift = self.drift.summary()

        status = "failed" if self._failure else "ended"
        reason = self._failure or self._end_reason
        if self.store is not None:
            try:
                billing = await self.store.end_session(self.config.session_id, self._failure)
                log.info(
                    "session settled",
                    extra={
                        "session_id": self.config.session_id,
                        "billed_language_seconds": billing.language_seconds,
                    },
                )
            except LookupError:
                log.warning("session already ended", extra={"session_id": self.config.session_id})

        log.info(
            "session summary",
            extra={
                "session_id": self.config.session_id,
                "status": status,
                "reason": reason,
                "chunks": self._chunks,
                "latency": latency,
                "backlog": backlog,
                "drift": drift,
            },
        )
        return SessionOutcome(
            session_id=self.config.session_id,
            status=status,
            chunks=self._chunks,
            failure_reason=reason,
            latency=latency,
            backlog=backlog,
            drift=drift,
        )
