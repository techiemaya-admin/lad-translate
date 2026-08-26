"""
Listener join service.

Scan a QR code, pick a language, hear the translation. No app, no account, no
login. The whole audience path is these four endpoints plus a static page.

    GET  /s/{session_id}                    the join page
    GET  /api/sessions/{session_id}         event name and available languages
    POST /api/sessions/{session_id}/join    issue a token for one language
    POST /api/listeners/{listener_id}/leave record the departure

The page is served from here rather than a CDN, and so is the LiveKit client
bundle. Conference wifi routinely blocks or throttles third party origins, and
a join page that cannot fetch its own client library is a room full of people
hearing nothing.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import TenantContext
from ..db.pool import create_pool
from ..db.sessions import SessionStore
from ..db.tenancy import TenantResolver
from ..obs.log import get_logger
from .languages import describe
from .rooms import SOURCE_TRACK_NAME, RoomInspector
from .tokens import TokenIssuer

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class JoinRequest(BaseModel):
    """
    Body of a join request.

    Defined at module level on purpose. This module uses
    `from __future__ import annotations`, so FastAPI resolves the handler's
    annotations against module globals; a model defined inside create_app is
    invisible there and the body silently degrades to a query parameter.
    """

    language: str


def create_app(
    pool=None,
    issuer: TokenIssuer | None = None,
    control_schema: str | None = None,
    database_url: str | None = None,
):
    """
    Build the FastAPI application.

    Pass `pool` to supply your own (tests), or `database_url` to have one
    opened in the lifespan. Opening it in the lifespan matters: asyncpg binds a
    pool to the event loop that created it, so a pool built on a throwaway loop
    before uvicorn starts fails on first use with "another operation is in
    progress".
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_pool = False
        if app.state.pool is None and database_url:
            app.state.pool = await create_pool(database_url)
            owns_pool = True
        if app.state.pool is not None and app.state.control_schema:
            app.state.resolver = TenantResolver(app.state.pool, app.state.control_schema)
        try:
            yield
        finally:
            if owns_pool and app.state.pool is not None:
                await app.state.pool.close()

    app = FastAPI(
        title="LAD Live Translation", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.pool = pool
    app.state.issuer = issuer
    app.state.control_schema = control_schema or os.getenv("LAD_CONTROL_SCHEMA")
    app.state.resolver = None
    app.state.inspector = (
        # internal_url, not the advertised one: this call runs on the server
        # and must not be routed out through the TLS proxy.
        RoomInspector(issuer.internal_url, issuer._api_key, issuer._api_secret)
        if issuer is not None
        else None
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -------------------------------------------------------------------------

    async def _load_session(session_id: str):
        """
        Find a session and the store that owns it.

        The tenant is derived from the session row, not from the request. A
        listener has no credentials and no tenant, so anything they send about
        one is untrusted.
        """
        if app.state.pool is None:
            raise HTTPException(503, "database not configured")

        # Sessions live in per-tenant schemas, so finding one means asking each
        # active tenant. Fine for a handful of tenants; when that stops being
        # true, add a session_id -> tenant_id index to the control plane rather
        # than widening this scan.
        tenants = await app.state.pool.fetch(
            f"SELECT id::text, schema_name FROM {app.state.control_schema}.tenants WHERE is_active"
        )
        for tenant_id, schema in [(r[0], r[1]) for r in tenants]:
            context = TenantContext(
                tenant_id=tenant_id,
                database_url=os.getenv("LAD_DATABASE_URL", "unused"),
                schema=schema,
            )
            store = SessionStore(app.state.pool, context)
            session = await store.get_session(session_id)
            if session is not None:
                return store, session
        raise HTTPException(404, "session not found")

    # -------------------------------------------------------------------------

    async def _load_room(room_name: str):
        """
        Find the session currently live in a room.

        The reason room URLs exist. A session id changes every time the service
        restarts, so a QR code printed against one is dead the moment anything
        is redeployed. Codes go on badges and signage days before an event, and
        a link that only survives until the next restart is not a link.
        """
        if app.state.pool is None:
            raise HTTPException(503, "database not configured")

        tenants = await app.state.pool.fetch(
            f"SELECT id::text, schema_name FROM {app.state.control_schema}.tenants "
            "WHERE is_active"
        )
        for tenant_id, schema in [(r[0], r[1]) for r in tenants]:
            context = TenantContext(
                tenant_id=tenant_id,
                database_url=os.getenv("LAD_DATABASE_URL", "unused"),
                schema=schema,
            )
            store = SessionStore(app.state.pool, context)
            for row in await store.live_sessions():
                session = await store.get_session(row["session_id"])
                if session is not None and session["room_name"] == room_name:
                    return store, session
        raise HTTPException(404, f"no live session in room {room_name!r}")

    @app.get("/room/{room_name}")
    async def room_listen_page(room_name: str):
        """Stable listener URL. Survives restarts; a session id does not."""
        return FileResponse(STATIC_DIR / "join.html")

    @app.get("/room/{room_name}/speak")
    async def room_speak_page(room_name: str):
        return FileResponse(STATIC_DIR / "speak.html")

    @app.get("/api/rooms/{room_name}")
    async def room_info(room_name: str):
        _store, session = await _load_room(room_name)
        return await _describe(session)

    @app.post("/api/rooms/{room_name}/join")
    async def room_join(room_name: str, body: JoinRequest, request: Request):
        store, session = await _load_room(room_name)
        return await _issue_listener(store, session, body.language, _client_url(request))

    @app.post("/api/rooms/{room_name}/speak")
    async def room_speak(room_name: str, request: Request):
        _store, session = await _load_room(room_name)
        return await _issue_speaker(session, _client_url(request))

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/s/{session_id}")
    async def join_page(session_id: str):
        """The page itself. Session detail is fetched by the page, not baked in."""
        return FileResponse(STATIC_DIR / "join.html")

    @app.get("/speak/{session_id}")
    async def speak_page(session_id: str):
        """
        The speaker's own page: publishes a phone microphone as source-audio.

        A venue would never use this. Section 8 is explicit that the feed comes
        from an aux or matrix send carrying the speaker mic only, through a USB
        interface or Dante. This exists so the pipeline can be driven by a
        human voice with no hardware at all, which is the difference between
        demonstrating the system and describing it.
        """
        return FileResponse(STATIC_DIR / "speak.html")

    async def _issue_speaker(session, url: str | None = None) -> dict:
        """Publish-only token for the speaker. Shared by both URL shapes."""
        if app.state.issuer is None:
            raise HTTPException(503, "LiveKit not configured")
        if session["status"] not in ("starting", "live"):
            raise HTTPException(410, "this session has ended")

        # Refuse a second speaker rather than let two people talk over each
        # other into one transcript. The SFU would happily carry both.
        if app.state.inspector is not None:
            live = await app.state.inspector.published_languages(session["room_name"])
            if SOURCE_TRACK_NAME in live:
                raise HTTPException(
                    409,
                    "someone is already speaking in this session; only one "
                    "source track is supported",
                )

        sid = session["session_id"]
        token = app.state.issuer.for_publisher(
            session["room_name"], identity=f"speaker-{sid[:8]}"
        )
        log.info("speaker token issued", extra={"session_id": sid})
        return {
            "url": url or app.state.issuer.livekit_url,
            "token": token,
            "track_name": SOURCE_TRACK_NAME,
            "event_name": session["event_name"],
            "source_language": session["source_language"],
        }

    @app.post("/api/sessions/{session_id}/speak")
    async def speak(session_id: str, request: Request):
        _store, session = await _load_session(session_id)
        return await _issue_speaker(session, _client_url(request))

    async def _describe(session) -> dict:
        """Language list with live availability. Shared by both URL shapes."""
        if session["status"] not in ("starting", "live"):
            raise HTTPException(410, "this session has ended")

        # What the session was configured to offer, against what the room is
        # actually publishing. A language in the first but not the second gives
        # a listener a working connection and silence.
        live = set()
        if app.state.inspector is not None:
            live = await app.state.inspector.published_languages(session["room_name"])

        configured = list(session["target_languages"])
        missing = [c for c in configured if live and c not in live]
        if missing:
            log.warning(
                "configured languages are not being published",
                extra={
                    "session_id": session["session_id"],
                    "missing": missing,
                    "publishing": sorted(live),
                },
            )

        languages = []

        # The original, first in the list. A relay of the speaker's own track:
        # no transcription, no translation, no synthesis, so it has none of
        # their errors and none of their latency. Costs nothing extra, because
        # the track is in the room whether anyone listens to it or not, and it
        # is not billed as a language for the same reason.
        source = describe(session["source_language"])
        languages.append(
            {
                "code": source.code,
                "native": source.native,
                "english": "original audio",
                "rtl": source.rtl,
                "available": (not live) or (SOURCE_TRACK_NAME in live),
                "is_source": True,
            }
        )

        for code in configured:
            info = describe(code)
            languages.append(
                {
                    "code": info.code,
                    "native": info.native,
                    "english": info.english,
                    "rtl": info.rtl,
                    # Unknown (empty `live`) counts as available: a failed
                    # inspection must not hide every language on the page.
                    "available": (not live) or (code in live),
                    "is_source": False,
                }
            )
        return {
            "session_id": session["session_id"],
            "event_name": session["event_name"],
            "status": session["status"],
            "languages": languages,
        }

    @app.get("/api/sessions/{session_id}")
    async def session_info(session_id: str):
        _store, session = await _load_session(session_id)
        return await _describe(session)

    def _client_url(request: Request) -> str:
        """
        The LiveKit URL to hand this particular client.

        A page served over http://localhost is already a secure context, so a
        browser there may open ws:// to localhost and needs no certificate at
        all. Handing it the public wss address instead would force it through
        a proxy whose CA it does not trust, for no gain.

        Anything arriving on a non-loopback host is a real remote client and
        gets the public address.
        """
        host = (request.url.hostname or "").lower()
        if host in ("localhost", "127.0.0.1", "::1"):
            return app.state.issuer.internal_url
        return app.state.issuer.livekit_url

    async def _issue_listener(store, session, language: str, url: str | None = None) -> dict:
        """
        Register a listener and mint their token.

        Shared by the session-id and room URL shapes so the two cannot drift.
        """
        if app.state.issuer is None:
            raise HTTPException(503, "LiveKit not configured")
        if session["status"] not in ("starting", "live"):
            raise HTTPException(410, "this session has ended")

        sid = session["session_id"]
        is_source = language == session["source_language"]
        if not is_source and language not in session["target_languages"]:
            offered = [session["source_language"], *session["target_languages"]]
            raise HTTPException(
                400, f"{language!r} is not offered; available: {offered}"
            )

        listener_id = await store.listener_joined(sid, language)
        token = app.state.issuer.for_listener(
            room=session["room_name"], language=language, listener_id=listener_id
        )
        # The original is a different track, not lang-<code>.
        track_name = SOURCE_TRACK_NAME if is_source else f"lang-{language}"
        log.info(
            "listener joined",
            extra={"session_id": sid, "listener_id": listener_id, "language": language},
        )
        return {
            "listener_id": listener_id,
            # A page reached by room name never saw a session id, and needs
            # this to report the listener leaving.
            "session_id": sid,
            "language": language,
            "url": url or token.url,
            "token": token.token,
            "track_name": track_name,
            "is_source": is_source,
        }

    @app.post("/api/sessions/{session_id}/join")
    async def join(session_id: str, body: JoinRequest, request: Request):
        store, session = await _load_session(session_id)
        return await _issue_listener(store, session, body.language, _client_url(request))

    @app.post("/api/listeners/{listener_id}/leave")
    async def leave(listener_id: str, session_id: str):
        """
        Record a departure.

        Called with sendBeacon on page unload, which is best effort by
        definition: a phone that loses signal or is force quit never sends it.
        Listener counts from this table are therefore a floor, not a census.
        """
        store, _session = await _load_session(session_id)
        await store.listener_left(listener_id)
        return JSONResponse({"ok": True})

    return app
