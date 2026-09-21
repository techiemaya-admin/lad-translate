"""
The console's glossary: the words this venue keeps getting wrong.

Mounted on the console, so it is behind the same Google sign-in as everything
else. A rule here changes what an audience hears, which is not a door to leave
easier to open than the one in front of the transcript.

EVERY ROUTE ANSWERS 200 WITH A `reason` WHEN THERE IS NO DATABASE, like the
hardware output and transcript panels and for the same reason: "this box has
no database" is something an operator reads once, not an error to retry.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..corrections import MAX_RULES
from ..db.corrections import CorrectionStore, DuplicateCorrection
from ..db.sessions import SessionStore
from ..db.tenancy import SchemaError
from ..obs.log import get_logger
from . import sessions

log = get_logger(__name__)


class CorrectionBody(BaseModel):
    wrong: str = Field(min_length=1, max_length=200)
    right: str = Field(max_length=200)
    language: str = Field(min_length=2, max_length=8)
    room: str | None = None
    everywhere: bool = False
    """True pins the rule to the tenant rather than to one room."""


async def _store(request: Request) -> CorrectionStore:
    cfg = request.app.state.outputs
    if not cfg.configured:
        raise HTTPException(503, cfg.why_not)
    try:
        tenant = await cfg.tenant()
    except SchemaError as exc:
        raise HTTPException(503, str(exc)) from exc
    return CorrectionStore(cfg.pool, tenant)


def install(app: FastAPI, prefix: str) -> None:
    """Mount the correction routes. The sign-in middleware gates them."""

    @app.get(f"{prefix}/api/corrections")
    async def list_corrections(request: Request, room: str = "dubai-demo"):
        cfg = request.app.state.outputs
        shell = {"room": room, "rules": [], "reason": "", "limit": MAX_RULES}
        if not cfg.configured:
            shell["reason"] = cfg.why_not
            return shell
        try:
            store = await _store(request)
            shell["rules"] = [
                {
                    "id": r["correction_id"],
                    "language": r["language"],
                    "wrong": r["wrong_text"],
                    "right": r["right_text"],
                    "room": r["room_name"],
                    "active": r["is_active"],
                    "created_at": str(r["created_at"]),
                    "created_by": r["created_by"],
                }
                for r in await store.list_rules(room)
            ]
        except HTTPException:
            raise
        except Exception as exc:
            if getattr(exc, "sqlstate", "") == "42P01":
                shell["reason"] = (
                    "this tenant's schema has no corrections table yet; "
                    "run the migrations."
                )
                return shell
            raise
        return shell

    @app.post(f"{prefix}/api/corrections")
    async def add_correction(body: CorrectionBody, request: Request):
        room = None if body.everywhere else body.room
        if room is not None:
            try:
                sessions.validate_room(room)
            except sessions.BadRoom as exc:
                raise HTTPException(400, str(exc)) from exc
        store = await _store(request)
        try:
            row = await store.add(
                language=body.language,
                wrong=body.wrong,
                right=body.right,
                room=room,
                created_by=getattr(request.state, "email", None),
            )
        except DuplicateCorrection as exc:
            # 409, not 400: the operator asked for something reasonable and
            # the answer is "that already exists", which the page can say.
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "id": row["correction_id"],
            "language": row["language"],
            "wrong": row["wrong_text"],
            "right": row["right_text"],
            "room": row["room_name"],
        }

    @app.delete(f"{prefix}/api/corrections/{{correction_id}}")
    async def delete_correction(correction_id: int, request: Request):
        store = await _store(request)
        if not await store.delete(correction_id):
            raise HTTPException(404, "no such correction")
        return {"deleted": correction_id}

    @app.post(f"{prefix}/api/corrections/replay")
    async def replay(request: Request, room: str = "dubai-demo"):
        """
        Apply every rule to the room's newest session's stored transcript.

        Separate from adding a rule, and deliberately not automatic: rewriting
        a stored transcript is a decision about the record, and an operator
        fixing a name for the next fifty minutes has not necessarily asked to
        rewrite the last ten.
        """
        try:
            sessions.validate_room(room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

        cfg = request.app.state.outputs
        store = await _store(request)
        tenant = await cfg.tenant()
        sess_store = SessionStore(cfg.pool, tenant)
        session = await sess_store.newest_session_in_room(room)
        if session is None:
            raise HTTPException(404, f"no session has run in {room} yet")

        rules = await store.load(room)
        if not rules:
            return {"rows_changed": 0, "words_changed": 0, "reason": "no corrections yet"}
        result = await store.replay(
            session["session_id"], rules, session["source_language"]
        )
        result["session_id"] = session["session_id"]
        return result
