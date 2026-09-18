"""
Reads and writes operator corrections in one tenant's schema.

The matching itself lives in lad_translate/corrections.py and knows nothing
about a database, because it is the part with the subtle behaviour and it
should be testable without one. This is only storage.
"""

from __future__ import annotations

from ..config import TenantContext
from ..corrections import MAX_RULES, Correction, Corrections, normalise
from ..obs.log import get_logger
from .tenancy import validate_schema

log = get_logger(__name__)


class DuplicateCorrection(ValueError):
    """A rule for those words in that language and scope already exists."""


class CorrectionStore:
    def __init__(self, pool, tenant: TenantContext) -> None:
        self._pool = pool
        self._tenant = tenant
        self._schema = validate_schema(tenant.schema)

    @property
    def tenant_id(self) -> str:
        return self._tenant.tenant_id

    async def list_rules(self, room: str | None = None) -> list[dict]:
        """
        Every rule the operator can see for a room: the tenant-wide ones and
        the ones pinned to that room. Newest first, because the rule just
        added is the one being checked.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT correction_id, language, wrong_text, right_text,
                           room_name, created_at, created_by
                      FROM {self._schema}.corrections
                     WHERE tenant_id = $1
                       AND ($2::text IS NULL OR room_name IS NULL OR room_name = $2)
                  ORDER BY created_at DESC, correction_id DESC""",
                self.tenant_id,
                room,
            )
        return [dict(r) for r in rows]

    async def load(self, room: str | None = None) -> Corrections:
        """The compiled rule set a session applies. This is the hot path."""
        rows = await self.list_rules(room)
        return Corrections(
            [Correction(r["wrong_text"], r["right_text"], r["language"]) for r in rows]
        )

    async def add(
        self,
        language: str,
        wrong: str,
        right: str,
        room: str | None = None,
        created_by: str | None = None,
    ) -> dict:
        wrong, right, language = normalise(wrong), normalise(right), language.strip()
        # Construct one first: the rules about blank text live in the matcher
        # and must not be restated here, where they would drift.
        Correction(wrong, right, language)

        if await self.count() >= MAX_RULES:
            raise ValueError(
                f"this tenant already has {MAX_RULES} corrections, which is the "
                "limit; delete rules that never fire before adding more"
            )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO {self._schema}.corrections
                        (tenant_id, language, wrong_text, right_text, room_name, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                    RETURNING correction_id, language, wrong_text, right_text,
                              room_name, created_at, created_by""",
                self.tenant_id, language, wrong, right, room, created_by,
            )
        if row is None:
            raise DuplicateCorrection(
                f"a correction for {wrong!r} in {language} already exists here"
            )
        log.info(
            "correction added",
            extra={"language": language, "room": room, "corrected": wrong},
        )
        return dict(row)

    async def delete(self, correction_id: int) -> bool:
        async with self._pool.acquire() as conn:
            done = await conn.execute(
                f"DELETE FROM {self._schema}.corrections "
                "WHERE correction_id = $1 AND tenant_id = $2",
                correction_id,
                self.tenant_id,
            )
        return done.endswith("1")

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT count(*) FROM {self._schema}.corrections WHERE tenant_id = $1",
                self.tenant_id,
            )

    async def replay(
        self, session_id: str, rules: Corrections, source_language: str
    ) -> dict:
        """
        Apply the rules to a session's ALREADY STORED transcript.

        The audience heard what they heard - this cannot un-say it, and does
        not pretend to. What it fixes is the record: the transcript panel, the
        downloads, the subtitles someone cuts onto the recording afterwards.

        Source rules are applied to source_text and target rules to
        translated_text, matching what the live path does, so replaying over a
        session produces the same words a session with those rules would have
        produced.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT transcript_id, language, source_text, translated_text
                      FROM {self._schema}.session_transcripts
                     WHERE session_id = $1 AND tenant_id = $2""",
                session_id,
                self.tenant_id,
            )
            updates, words = [], 0
            for row in rows:
                source, hit_a = rules.apply(row["source_text"], source_language)
                target, hit_b = rules.apply(row["translated_text"] or "", row["language"])
                if hit_a or hit_b:
                    words += hit_a + hit_b
                    updates.append(
                        (row["transcript_id"], source, target or row["translated_text"])
                    )
            if updates:
                await conn.executemany(
                    f"""UPDATE {self._schema}.session_transcripts
                           SET source_text = $2, translated_text = $3
                         WHERE transcript_id = $1""",
                    updates,
                )
        log.info(
            "corrections replayed over a stored transcript",
            extra={"session_id": session_id, "rows": len(updates), "words": words},
        )
        return {"rows_changed": len(updates), "words_changed": words, "rows_seen": len(rows)}
