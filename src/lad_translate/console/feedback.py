"""
Thumbs on the live transcript, and the Learnings panel that shows what they taught.

Mounted on the console behind the same sign-in as the transcript itself. A
thumbs-down here can write a correction rule, which changes what an audience
hears, so it gets no easier door than the corrections panel has.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..db.feedback import RATINGS, FeedbackStore
from ..db.sessions import SessionStore
from ..db.tenancy import SchemaError
from ..obs.log import get_logger
from . import sessions

log = get_logger(__name__)


class FeedbackBody(BaseModel):
    room: str
    chunk_id: int = Field(ge=0)
    language: str = Field(min_length=2, max_length=8)
    rating: str
    produced_text: str = Field(max_length=2000)
    expected_text: str | None = Field(default=None, max_length=2000)
    note: str | None = Field(default=None, max_length=500)


class ActiveBody(BaseModel):
    active: bool


async def _stores(request: Request) -> tuple[FeedbackStore, SessionStore]:
    cfg = request.app.state.outputs
    if not cfg.configured:
        raise HTTPException(503, cfg.why_not)
    try:
        tenant = await cfg.tenant()
    except SchemaError as exc:
        raise HTTPException(503, str(exc)) from exc
    return FeedbackStore(cfg.pool, tenant), SessionStore(cfg.pool, tenant)


async def _newest_session(sessions_store: SessionStore, room: str) -> dict:
    try:
        sessions.validate_room(room)
    except sessions.BadRoom as exc:
        raise HTTPException(400, str(exc)) from exc
    session = await sessions_store.newest_session_in_room(room)
    if session is None:
        raise HTTPException(404, f"no session has run in {room} yet")
    return session


def _public(row: dict) -> dict:
    return {
        "id": row["feedback_id"],
        "chunk_id": row["chunk_id"],
        "language": row["language"],
        "rating": row["rating"],
        "produced": row["produced_text"],
        "expected": row["expected_text"],
        "note": row["note"],
        "learned": row["correction_id"] is not None,
        "learned_reason": row["learned_reason"],
        "rule": (
            {"wrong": row["wrong_text"], "right": row["right_text"]}
            if row.get("wrong_text") is not None
            else None
        ),
        "active": row["is_active"],
        "created_by": row["created_by"],
        "updated_at": str(row["updated_at"]),
    }


def install(app: FastAPI, prefix: str) -> None:
    """Mount the feedback routes. The sign-in middleware gates them."""

    @app.get(f"{prefix}/api/transcript/feedback")
    async def list_feedback(request: Request, room: str = "dubai-demo"):
        """
        Everything the operator has said about the room's newest session,
        with the per-language thumbs that make the quality signal visible.
        200 with a `reason` when there is nothing to show, like the panels
        around it.
        """
        cfg = request.app.state.outputs
        shell = {"room": room, "session_id": None, "items": [], "stats": [], "reason": ""}
        if not cfg.configured:
            shell["reason"] = cfg.why_not
            return shell
        try:
            store, sessions_store = await _stores(request)
            session = await _newest_session(sessions_store, room)
        except HTTPException as exc:
            if exc.status_code == 404:
                shell["reason"] = str(exc.detail)
                return shell
            raise
        shell["session_id"] = session["session_id"]
        try:
            shell["items"] = [_public(r) for r in await store.list_for_session(session["session_id"])]
            shell["stats"] = [dict(s) for s in await store.stats_for_session(session["session_id"])]
        except Exception as exc:
            if getattr(exc, "sqlstate", "") == "42P01":
                shell["reason"] = "this tenant's schema has no feedback table yet; run the migrations."
                return shell
            raise
        return shell

    @app.post(f"{prefix}/api/transcript/feedback")
    async def give_feedback(body: FeedbackBody, request: Request):
        if body.rating not in RATINGS:
            raise HTTPException(400, f"rating must be one of {RATINGS}")
        store, sessions_store = await _stores(request)
        session = await _newest_session(sessions_store, body.room)
        try:
            row = await store.give(
                session_id=session["session_id"],
                chunk_id=body.chunk_id,
                language=body.language,
                rating=body.rating,
                produced_text=body.produced_text,
                expected_text=body.expected_text,
                note=body.note,
                # The dictionary is the VENUE's, not the room's. A name taught
                # in one room is known in every room from then on, the way a
                # phone's personal dictionary does not relearn a word per
                # conversation. Room-scoped rules still exist; they are the
                # exception an operator writes by hand.
                room=None,
                created_by=getattr(request.state, "email", None),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _public({**row, "wrong_text": None, "right_text": None}) | {
            # The rule text is not joined on the insert path; say what was
            # learned from the reason, which already names it.
            "rule": None,
        }

    @app.post(f"{prefix}/api/transcript/feedback/{{feedback_id}}/active")
    async def set_active(feedback_id: int, body: ActiveBody, request: Request):
        store, _ = await _stores(request)
        if not await store.set_active(feedback_id, body.active):
            raise HTTPException(404, "no such feedback")
        return {"id": feedback_id, "active": body.active}

    @app.delete(f"{prefix}/api/transcript/feedback/{{feedback_id}}")
    async def delete_feedback(feedback_id: int, request: Request):
        store, _ = await _stores(request)
        if not await store.delete(feedback_id):
            raise HTTPException(404, "no such feedback")
        return {"deleted": feedback_id}
