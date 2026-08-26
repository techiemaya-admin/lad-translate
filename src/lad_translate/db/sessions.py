"""
Session storage.

Every statement filters on tenant_id, including the ones where the primary key
alone would find the row. That is not redundancy. A session_id is enough to
identify a session, so a bug that passes the wrong tenant_id would happily
return another tenant's data. With the filter, the same bug returns nothing.

The schema is validated once, in the constructor, and interpolated from there.
It never comes from an environment default or a module constant.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..config import SessionConfig, TenantContext
from ..obs.log import get_logger
from .tenancy import validate_schema

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionBilling:
    """
    What a session is charged for.

    Duration multiplied by active language count. Not per call, and it does not
    key on call_log_id: VOAG's recordVoiceCallUsage() is the wrong shape for an
    event that runs for hours with a fixed set of languages.

    Settled at close and stored, never recomputed from live config afterwards.
    """

    session_id: str
    tenant_id: str
    seconds: int
    language_count: int

    @property
    def language_seconds(self) -> int:
        """The billable quantity."""
        return self.seconds * self.language_count


@dataclass(frozen=True, slots=True)
class TranscriptRow:
    chunk_id: int
    language: str
    source_text: str
    translated_text: str | None
    t_audio_start: float
    t_audio_end: float
    latency_s: float | None
    commit_reason: str | None
    revised: bool


class SessionStore:
    """Reads and writes session data in one tenant's schema."""

    def __init__(self, pool, tenant: TenantContext) -> None:
        self._pool = pool
        self._tenant = tenant
        self._schema = validate_schema(tenant.schema)

    @property
    def tenant_id(self) -> str:
        return self._tenant.tenant_id

    @property
    def schema(self) -> str:
        return self._schema

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------

    async def create_session(self, config: SessionConfig, latency_credible: bool) -> str:
        if config.tenant.tenant_id != self.tenant_id:
            raise ValueError(
                f"session tenant {config.tenant.tenant_id} does not match store "
                f"tenant {self.tenant_id}"
            )
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.translation_sessions (
                session_id, tenant_id, event_name, room_name,
                source_language, target_languages, status,
                latency_credible, backend_stt, backend_mt, backend_tts
            ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, 'starting', $7, $8, $9, $10)
            """,
            config.session_id,
            self.tenant_id,
            config.event_name,
            config.room_name,
            config.source_language,
            config.target_codes,
            latency_credible,
            config.backends.stt,
            config.backends.mt,
            config.backends.tts,
        )
        log.info(
            "session created",
            extra={
                "session_id": config.session_id,
                "tenant_id": self.tenant_id,
                "languages": config.target_codes,
                "latency_credible": latency_credible,
            },
        )
        return config.session_id

    async def mark_live(self, session_id: str) -> None:
        await self._update_status(session_id, "live")

    async def end_session(
        self, session_id: str, failure_reason: str | None = None
    ) -> SessionBilling:
        """
        Close the session and settle billing.

        The quantity is computed here, from the row's own started_at and
        target_languages, and written down. Recomputing it later from config
        would let a configuration change alter an invoice already raised.
        """
        status = "failed" if failure_reason else "ended"
        row = await self._pool.fetchrow(
            f"""
            UPDATE {self._schema}.translation_sessions
               SET status = $3,
                   ended_at = now(),
                   failure_reason = $4,
                   billed_seconds = GREATEST(0, CEIL(EXTRACT(EPOCH FROM (now() - started_at)))::int),
                   billed_language_count = cardinality(target_languages)
             WHERE session_id = $1::uuid
               AND tenant_id = $2::uuid
               AND ended_at IS NULL
            RETURNING billed_seconds, billed_language_count
            """,
            session_id,
            self.tenant_id,
            status,
            failure_reason,
        )
        if row is None:
            raise LookupError(
                f"session {session_id} not found for tenant {self.tenant_id}, "
                "or already ended"
            )
        billing = SessionBilling(
            session_id=session_id,
            tenant_id=self.tenant_id,
            seconds=row[0],
            language_count=row[1],
        )
        log.info(
            "session ended",
            extra={
                "session_id": session_id,
                "tenant_id": self.tenant_id,
                "status": status,
                "billed_seconds": billing.seconds,
                "billed_language_count": billing.language_count,
                "billed_language_seconds": billing.language_seconds,
                "failure_reason": failure_reason,
            },
        )
        return billing

    async def _update_status(self, session_id: str, status: str) -> None:
        result = await self._pool.execute(
            f"""
            UPDATE {self._schema}.translation_sessions
               SET status = $3
             WHERE session_id = $1::uuid AND tenant_id = $2::uuid
            """,
            session_id,
            self.tenant_id,
            status,
        )
        if result.endswith(" 0"):
            raise LookupError(f"session {session_id} not found for tenant {self.tenant_id}")

    async def get_session(self, session_id: str):
        return await self._pool.fetchrow(
            f"""
            SELECT session_id::text, tenant_id::text, event_name, room_name,
                   source_language, target_languages, status,
                   started_at, ended_at, failure_reason,
                   billed_seconds, billed_language_count, latency_credible
              FROM {self._schema}.translation_sessions
             WHERE session_id = $1::uuid AND tenant_id = $2::uuid
            """,
            session_id,
            self.tenant_id,
        )

    async def live_sessions(self) -> list:
        return await self._pool.fetch(
            f"""
            SELECT session_id::text, event_name, started_at, target_languages
              FROM {self._schema}.translation_sessions
             WHERE tenant_id = $1::uuid AND status IN ('starting', 'live')
             ORDER BY started_at DESC
            """,
            self.tenant_id,
        )

    # -------------------------------------------------------------------------
    # Listeners
    # -------------------------------------------------------------------------

    async def listener_joined(self, session_id: str, language: str) -> str:
        listener_id = str(uuid.uuid4())
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.session_listeners
                (listener_id, session_id, tenant_id, language)
            VALUES ($1::uuid, $2::uuid, $3::uuid, $4)
            """,
            listener_id,
            session_id,
            self.tenant_id,
            language,
        )
        return listener_id

    async def listener_left(self, listener_id: str) -> None:
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.session_listeners
               SET left_at = now()
             WHERE listener_id = $1::uuid AND tenant_id = $2::uuid AND left_at IS NULL
            """,
            listener_id,
            self.tenant_id,
        )

    async def active_listeners(self, session_id: str) -> dict[str, int]:
        """Live listener count per language. Drives the room load picture."""
        rows = await self._pool.fetch(
            f"""
            SELECT language, count(*)::int
              FROM {self._schema}.session_listeners
             WHERE tenant_id = $1::uuid AND session_id = $2::uuid AND left_at IS NULL
             GROUP BY language
            """,
            self.tenant_id,
            session_id,
        )
        return {row[0]: row[1] for row in rows}

    # -------------------------------------------------------------------------
    # Transcripts
    # -------------------------------------------------------------------------

    async def record_transcript(self, session_id: str, row: TranscriptRow) -> None:
        """
        Store one chunk in one language.

        Upsert rather than insert: a retried publish must not fail the session
        on a unique violation, and the second attempt carries the better data.
        """
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.session_transcripts (
                session_id, tenant_id, chunk_id, language,
                source_text, translated_text,
                t_audio_start, t_audio_end, latency_s, commit_reason, revised
            ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (session_id, chunk_id, language) DO UPDATE
               SET translated_text = EXCLUDED.translated_text,
                   latency_s = EXCLUDED.latency_s,
                   revised = EXCLUDED.revised
            """,
            session_id,
            self.tenant_id,
            row.chunk_id,
            row.language,
            row.source_text,
            row.translated_text,
            row.t_audio_start,
            row.t_audio_end,
            row.latency_s,
            row.commit_reason,
            row.revised,
        )

    async def transcript(self, session_id: str, language: str) -> list:
        return await self._pool.fetch(
            f"""
            SELECT chunk_id, source_text, translated_text,
                   t_audio_start, t_audio_end, latency_s, commit_reason, revised
              FROM {self._schema}.session_transcripts
             WHERE tenant_id = $1::uuid AND session_id = $2::uuid AND language = $3
             ORDER BY chunk_id
            """,
            self.tenant_id,
            session_id,
            language,
        )
