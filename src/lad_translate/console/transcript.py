"""
The live transcript, for the console: what was said and what went out.

The one thing an operator cannot check from the console without borrowing a
phone. The status tiles say a session is healthy; they cannot say the
Arabic is coming out as Quranic exegesis, which is a thing that has
happened here and was found only because someone read the output. So the
console shows the source line and every language's line beside it, with
the latency each one actually achieved.

It reads the database rather than the journal. The session already writes a
row per chunk per language - source_text, translated_text, latency_s - and
the console already holds a pool and a tenant for the hardware output
panel. The journal carries no text, so parsing it would mean changing what
the session logs and shipping transcripts through the system log, which is
the wrong place for a talk's content.

POLLING, WITH A HIGH-WATER MARK. The page sends the highest chunk_id it has
and gets only what is newer. A forty-minute keynote is thousands of rows;
sending all of them every two seconds to draw a panel nobody has scrolled
is how a console becomes the reason a box is busy.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from ..db.sessions import SessionStore
from ..db.tenancy import SchemaError
from ..obs.log import get_logger
from . import sessions

log = get_logger(__name__)

MAX_ROWS = 300
"""Rows per poll, across all languages. Three languages make that a hundred
chunks - far more than a page shows, and a hard ceiling on the query."""


def group_by_chunk(rows) -> list[dict]:
    """
    One entry per chunk: the source line, and each language beside it.

    The source text is the same on every row of a chunk, which is what makes
    this collapse safe. A chunk whose translation failed still appears, with
    that language absent rather than the whole line missing - a phrase that
    reached three languages and not the fourth is exactly what an operator
    needs to see.
    """
    chunks: dict[int, dict] = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        entry = chunks.get(chunk_id)
        if entry is None:
            entry = chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "source": row["source_text"],
                "t_audio_start": round(float(row["t_audio_start"]), 2),
                "t_audio_end": round(float(row["t_audio_end"]), 2),
                "languages": {},
            }
        entry["languages"][row["language"]] = {
            "text": row["translated_text"],
            "latency_s": round(float(row["latency_s"]), 2) if row["latency_s"] is not None else None,
        }
    return [chunks[k] for k in sorted(chunks)]


def install(app: FastAPI, prefix: str) -> None:
    """Mount the transcript route. The sign-in middleware gates it."""

    @app.get(f"{prefix}/api/transcript")
    async def transcript(request: Request, room: str = "dubai-demo", after: int = -1):
        """
        Chunks newer than `after`, for the room's newest session.

        200 in every state, like the hardware output overview and for the
        same reason: "this box has no database" is something the operator
        needs to read once, not an error to retry. `reason` says which
        state, `chunks` is empty in all of them.
        """
        try:
            sessions.validate_room(room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

        cfg = request.app.state.outputs  # the console's pool and tenant
        shell = {
            "room": room,
            "session_id": None,
            "event_name": None,
            "status": None,
            "chunks": [],
            "reason": "",
        }
        if not cfg.configured:
            shell["reason"] = cfg.why_not
            return shell

        try:
            tenant = await cfg.tenant()
        except SchemaError as exc:
            shell["reason"] = str(exc)
            return shell

        store = SessionStore(cfg.pool, tenant)
        try:
            session = await store.newest_session_in_room(room)
            if session is None:
                shell["reason"] = f"no session has run in {room} yet."
                return shell
            shell["session_id"] = session["session_id"]
            shell["event_name"] = session["event_name"]
            shell["status"] = session["status"]
            rows = await store.recent_transcript(
                session["session_id"], after_chunk_id=after, limit=MAX_ROWS
            )
        except Exception as exc:
            # 42P01: the tenant schema predates the transcript tables. Same
            # shape of answer as the outputs panel gives.
            if getattr(exc, "sqlstate", "") == "42P01":
                shell["reason"] = "this tenant's schema has no transcript tables yet."
                return shell
            raise

        shell["chunks"] = group_by_chunk(rows)
        return shell
