"""
Reads and writes transcript feedback in one tenant's schema.

A thumbs-down that teaches something writes TWO rows - the feedback and the
correction it became - and they have to land together: a feedback row that
claims to have learned a rule which does not exist is the kind of lie an
operator finds out about mid-talk.
"""

from __future__ import annotations

from ..config import TenantContext
from ..corrections import Correction, normalise
from ..feedback import derive_rule
from ..obs.log import get_logger
from .tenancy import validate_schema

log = get_logger(__name__)

RATINGS = ("like", "dislike")


class FeedbackStore:
    def __init__(self, pool, tenant: TenantContext) -> None:
        self._pool = pool
        self._tenant = tenant
        self._schema = validate_schema(tenant.schema)

    @property
    def tenant_id(self) -> str:
        return self._tenant.tenant_id

    async def give(
        self,
        *,
        session_id: str,
        chunk_id: int,
        language: str,
        rating: str,
        produced_text: str,
        expected_text: str | None,
        note: str | None,
        room: str | None,
        created_by: str | None,
    ) -> dict:
        """
        Record a thumbs on one line, learning a rule from it when that is safe.

        One thumbs per line per language: a second opinion replaces the first.
        Replacing a thumbs-down that HAD learned a rule retires that rule, so an
        operator who changes their mind is not left with a correction they no
        longer stand behind and cannot see from the transcript.
        """
        if rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS}")
        language = language.strip()
        expected = normalise(expected_text or "")
        derived = (
            derive_rule(produced_text, expected, language)
            if rating == "dislike" and expected
            else None
        )

        async with self._pool.acquire() as conn, conn.transaction():
            previous = await conn.fetchrow(
                f"""SELECT feedback_id, correction_id
                      FROM {self._schema}.transcript_feedback
                     WHERE session_id = $1 AND chunk_id = $2 AND language = $3
                       AND tenant_id = $4""",
                session_id, chunk_id, language, self.tenant_id,
            )
            if previous is not None and previous["correction_id"] is not None:
                await conn.execute(
                    f"DELETE FROM {self._schema}.corrections "
                    "WHERE correction_id = $1 AND tenant_id = $2",
                    previous["correction_id"], self.tenant_id,
                )

            correction_id = None
            reason = ""
            if derived is not None:
                reason = derived.reason
                if derived.rule is not None:
                    correction_id = await self._add_rule(conn, derived.rule, room, created_by)
                    if correction_id is None:
                        reason = (
                            f"'{derived.rule.wrong}' already has a correction here; "
                            "this feedback is kept as an example"
                        )
            elif rating == "dislike":
                reason = "no 'should have been' given, so nothing was learned"

            row = await conn.fetchrow(
                f"""INSERT INTO {self._schema}.transcript_feedback
                        (tenant_id, session_id, chunk_id, language, rating,
                         produced_text, expected_text, note, correction_id,
                         learned_reason, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (session_id, chunk_id, language) DO UPDATE SET
                        rating         = EXCLUDED.rating,
                        produced_text  = EXCLUDED.produced_text,
                        expected_text  = EXCLUDED.expected_text,
                        note           = EXCLUDED.note,
                        correction_id  = EXCLUDED.correction_id,
                        learned_reason = EXCLUDED.learned_reason,
                        is_active      = true,
                        updated_at     = now(),
                        created_by     = EXCLUDED.created_by
                    RETURNING *""",
                self.tenant_id, session_id, chunk_id, language, rating,
                produced_text, expected or None, (note or "").strip() or None,
                correction_id, reason, created_by,
            )
        log.info(
            "transcript feedback",
            extra={
                "session_id": session_id, "chunk_id": chunk_id, "lang": language,
                "rating": rating, "learned": correction_id is not None,
            },
        )
        return dict(row)

    async def _add_rule(self, conn, rule: Correction, room: str | None, who: str | None):
        row = await conn.fetchrow(
            f"""INSERT INTO {self._schema}.corrections
                    (tenant_id, language, wrong_text, right_text, room_name, created_by)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING
                RETURNING correction_id""",
            self.tenant_id, rule.language, rule.wrong, rule.right, room, who,
        )
        return row["correction_id"] if row else None

    async def list_for_session(self, session_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT f.*, c.wrong_text, c.right_text,
                           c.is_active AS rule_active
                      FROM {self._schema}.transcript_feedback f
                 LEFT JOIN {self._schema}.corrections c USING (correction_id)
                     WHERE f.session_id = $1 AND f.tenant_id = $2
                  ORDER BY f.updated_at DESC, f.feedback_id DESC""",
                session_id, self.tenant_id,
            )
        return [dict(r) for r in rows]

    async def stats_for_session(self, session_id: str) -> list[dict]:
        """
        Thumbs per language. This is the quality signal a thumbs-UP exists
        for: a language with a rising dislike share is one whose translator
        the venue should not trust for the next event.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT language,
                           count(*) FILTER (WHERE rating = 'like')    AS likes,
                           count(*) FILTER (WHERE rating = 'dislike') AS dislikes,
                           count(*) FILTER (WHERE correction_id IS NOT NULL) AS learned
                      FROM {self._schema}.transcript_feedback
                     WHERE session_id = $1 AND tenant_id = $2
                  GROUP BY language ORDER BY language""",
                session_id, self.tenant_id,
            )
        return [dict(r) for r in rows]

    async def set_active(self, feedback_id: int, active: bool) -> bool:
        """
        Turn a learning off or on. Switches the RULE too, because the rule is
        the learning: a feedback row marked active whose rule is off would be
        a panel that says one thing and a pipeline that does another.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"""UPDATE {self._schema}.transcript_feedback
                       SET is_active = $3, updated_at = now()
                     WHERE feedback_id = $1 AND tenant_id = $2
                 RETURNING correction_id""",
                feedback_id, self.tenant_id, active,
            )
            if row is None:
                return False
            if row["correction_id"] is not None:
                await conn.execute(
                    f"UPDATE {self._schema}.corrections SET is_active = $3 "
                    "WHERE correction_id = $1 AND tenant_id = $2",
                    row["correction_id"], self.tenant_id, active,
                )
        return True

    async def delete(self, feedback_id: int) -> bool:
        """Deleting feedback deletes the rule it taught. The reverse is not true."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"""DELETE FROM {self._schema}.transcript_feedback
                     WHERE feedback_id = $1 AND tenant_id = $2
                 RETURNING correction_id""",
                feedback_id, self.tenant_id,
            )
            if row is None:
                return False
            if row["correction_id"] is not None:
                await conn.execute(
                    f"DELETE FROM {self._schema}.corrections "
                    "WHERE correction_id = $1 AND tenant_id = $2",
                    row["correction_id"], self.tenant_id,
                )
        return True
