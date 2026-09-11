"""
Operator console.

Served from the VM behind Google sign-in, NOT from the Cloud Run join
service. That service is deliberately --allow-unauthenticated, because a
listener scans a QR code and has no credentials; a surface that can restart
sessions and change models cannot share that door.

It also has to reach systemd, which is local to this box and unreachable from
Cloud Run.
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from ..db.tenancy import TenantResolver
from ..obs.log import get_logger
from . import auth, env, outputs, sessions
from .presets import BY_KEY, PRESETS

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "api" / "static"

PREFIX = "/console"
"""
Every route carries it, and Caddy passes the prefix through untouched.

The alternative - strip it at the proxy and serve from the root - is what broke
the first deploy: the page's absolute asset paths landed outside the protected
route and the browser re-prompted for credentials on every one.
"""


class ApplyRequest(BaseModel):
    room: str = Field(min_length=1, max_length=63)
    preset: str | None = None
    settings: dict[str, str] = Field(default_factory=dict)
    restart: bool = True


def create_app(
    public_base: str,
    env_path: Path | None = None,
    auth_config: auth.Config | None = None,
    database_url: str | None = None,
    control_schema: str | None = None,
    tenant: str | None = None,
    pool=None,
) -> FastAPI:
    """
    `public_base` is the URL a PHONE reaches the join service on - the Cloud Run
    address, not this box. The console runs on the SFU host; the QR codes it
    prints must not point at it.

    The database is optional. Without one the console still does everything it
    did before this - presets, settings, restarts, QR codes - and the hardware
    output panel says it is not configured rather than pretending the venue
    owns no devices. Pass `pool` to hand one in (tests), or `database_url` to
    have one opened in the lifespan: asyncpg binds a pool to the loop that
    created it, and uvicorn's loop does not exist yet when this runs.
    """
    database_url = database_url or os.getenv("LAD_DATABASE_URL", "")
    control_schema = control_schema or os.getenv("LAD_CONTROL_SCHEMA", "")
    tenant = tenant or os.getenv("LAD_TRANSLATE_TENANT", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg: outputs.OutputsConfig = app.state.outputs
        owns_pool = False
        if cfg.pool is None and database_url and control_schema:
            from ..db.pool import create_pool

            cfg.pool = await create_pool(database_url)
            cfg.resolver = TenantResolver(cfg.pool, control_schema)
            owns_pool = True
            log.info(
                "console database ready",
                extra={"control_schema": control_schema, "tenant": tenant},
            )
        try:
            yield
        finally:
            if owns_pool and cfg.pool is not None:
                await cfg.pool.close()

    app = FastAPI(
        title="LAD Live Translation - console",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.public_base = public_base.rstrip("/")
    app.state.env_path = env_path or env.DEFAULT_PATH
    # Built eagerly when a pool is handed in, for the same reason api/admin.py
    # does: a test client can drive the app without ever running the lifespan.
    app.state.outputs = outputs.OutputsConfig(
        pool=pool,
        resolver=TenantResolver(pool, control_schema) if pool is not None and control_schema else None,
        tenant_slug=tenant,
        database_url=database_url,
    )

    # Everything lives under /console, and Caddy does NOT strip the prefix.
    #
    # It used to strip it, and the page asked the browser for /static/console.css
    # and /api/presets - absolute paths that then fell outside the console route
    # entirely. Every one came back 401, so the browser re-prompted for
    # credentials on each asset and the console looked like a login that would
    # not take a correct password.
    #
    # Relative paths would have fixed the assets and left a trailing-slash trap:
    # served at /console they resolve against /, served at /console/ against
    # /console/. Owning the prefix removes the class of bug rather than the
    # instance.
    app.state.auth = auth_config or auth.config_from_env()

    # Paths that must work before anyone is signed in. Everything else is gated.
    # An allowlist, not a denylist: a new route is protected by default, which
    # is the direction a mistake should fall.
    OPEN = {
        f"{PREFIX}/auth/login",
        f"{PREFIX}/auth/callback",
        f"{PREFIX}/auth/denied",
        f"{PREFIX}/health",
    }

    class RequireGoogleSignIn(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            cfg: auth.Config = request.app.state.auth
            path = request.url.path

            if not cfg.enabled:
                # No client configured means no console, not an open console.
                # The Caddy incident earlier today came from treating missing
                # config as permission to carry on; this fails the other way.
                return Response(
                    "The console is not configured for sign-in. See "
                    "deploy/README.md.",
                    status_code=503,
                    media_type="text/plain",
                )
            if path in OPEN or path.startswith(f"{PREFIX}/static/"):
                return await call_next(request)

            email = auth.read_session(
                request.cookies.get(auth.COOKIE), cfg.session_secret
            )
            if not email:
                # An API call gets 401 so the page can say so; a navigation gets
                # sent to Google. Redirecting a fetch would hand the caller
                # Google's HTML and look like a parsing bug.
                if path.startswith(f"{PREFIX}/api/"):
                    return Response(
                        json.dumps({"detail": "sign in required"}),
                        status_code=401,
                        media_type="application/json",
                    )
                return RedirectResponse(f"{PREFIX}/auth/login", status_code=302)

            request.state.email = email
            return await call_next(request)

    app.add_middleware(RequireGoogleSignIn)

    app.mount(f"{PREFIX}/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get(f"{PREFIX}/health")
    async def health():
        return {"ok": True}

    @app.get(PREFIX)
    @app.get(f"{PREFIX}/")
    async def page():
        return FileResponse(STATIC_DIR / "console.html")

    @app.get(f"{PREFIX}/api/presets")
    async def presets():
        return {"presets": [p.as_dict() for p in PRESETS]}

    @app.get(f"{PREFIX}/api/settings")
    async def settings():
        current = env.read(app.state.env_path)
        return {
            "settings": {k: v for k, v in current.items() if k in env.EDITABLE},
            "editable": sorted(env.EDITABLE),
            "public_base": app.state.public_base,
        }

    @app.get(f"{PREFIX}/api/status")
    async def status(room: str = "dubai-demo"):
        try:
            return (await sessions.status(room)).__dict__
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post(f"{PREFIX}/api/apply")
    async def apply(body: ApplyRequest):
        try:
            sessions.validate_room(body.room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

        updates: dict[str, str] = {}
        if body.preset:
            preset = BY_KEY.get(body.preset)
            if preset is None:
                raise HTTPException(400, f"unknown preset {body.preset!r}")
            updates.update(
                {
                    "STT_BACKEND": preset.stt_backend,
                    "LAD_TRANSLATE_STT_MODEL": preset.model,
                    "LAD_TRANSLATE_EMIT_INTERVAL": str(preset.emit_interval),
                    "LAD_TRANSLATE_WINDOW": str(preset.window),
                    "LAD_TRANSLATE_LOOKAHEAD": preset.lookahead,
                }
            )
        # Raw values win over the preset, so "preset then adjust one field" does
        # what it looks like it does.
        updates.update(body.settings)

        try:
            changed = env.write(updates, app.state.env_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        restarted = False
        if body.restart:
            try:
                await sessions.restart(body.room)
                restarted = True
            except RuntimeError as exc:
                raise HTTPException(500, str(exc)) from exc

        log.info(
            "console applied settings",
            extra={"room": body.room, "preset": body.preset, "changed": changed},
        )
        return {"changed": changed, "restarted": restarted}

    @app.post(f"{PREFIX}/api/stop")
    async def stop(body: ApplyRequest):
        try:
            await sessions.stop(body.room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"stopped": True}

    def _qr_png(url: str) -> bytes:
        import qrcode

        code = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        code.add_data(url)
        code.make(fit=True)
        buf = io.BytesIO()
        code.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
        return buf.getvalue()

    def _room_urls(room: str) -> dict[str, str]:
        base = f"{app.state.public_base}/room/{room}"
        return {"listen": base, "speak": f"{base}/speak"}

    @app.get(f"{PREFIX}/api/qr.json")
    async def qr_json(room: str):
        """
        Both codes as data URIs, in one authenticated fetch.

        The page used to point two <img> tags at this API. The stylesheet and
        script survive in the browser cache, so they are fetched once and never
        challenged again - but these carry no-store and a cache-buster, so they
        hit the network fresh every time and the browser put up a second sign-in
        dialog over a page that had already loaded. It looked like a login that
        would not stay logged in.

        A data URI is not a request, so there is nothing left to challenge. It
        also means the codes render on a venue network that blocks everything
        except the page itself.
        """
        try:
            sessions.validate_room(room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

        urls = _room_urls(room)
        return {
            "urls": urls,
            "images": {
                kind: "data:image/png;base64," + base64.b64encode(_qr_png(url)).decode()
                for kind, url in urls.items()
            },
        }

    @app.get(f"{PREFIX}/api/qr")
    async def qr(room: str, kind: str = "listen"):
        """
        The QR code, as a PNG.

        Room URLs, never session URLs: a session id dies with the session, and
        a printed code built on one is a wall of paper pointing at a 404 the
        first time a worker restarts.
        """
        try:
            sessions.validate_room(room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc
        if kind not in ("listen", "speak"):
            raise HTTPException(400, "kind must be listen or speak")

        url = _room_urls(room)[kind]
        return Response(
            content=_qr_png(url),
            media_type="image/png",
            headers={"X-Encoded-Url": url, "Cache-Control": "no-store"},
        )

    # --- sign in -------------------------------------------------------------

    @app.get(f"{PREFIX}/auth/login")
    async def login(request: Request):
        cfg: auth.Config = request.app.state.auth
        state, nonce = auth.new_state(), auth.new_state()
        response = RedirectResponse(auth.authorize_url(cfg, state, nonce), status_code=302)
        # state and nonce ride in short-lived cookies rather than server memory,
        # so the console survives a restart mid-sign-in and needs no store.
        for name, value in (("lad_oauth_state", state), ("lad_oauth_nonce", nonce)):
            response.set_cookie(
                name, value, max_age=600, httponly=True, secure=True, samesite="lax"
            )
        return response

    @app.get(f"{PREFIX}/auth/callback")
    async def callback(request: Request, code: str = "", state: str = ""):
        cfg: auth.Config = request.app.state.auth
        expected = request.cookies.get("lad_oauth_state")
        nonce = request.cookies.get("lad_oauth_nonce")

        # compare_digest, and both halves must exist: a missing cookie and a
        # forged state should fail identically.
        if not code or not expected or not secrets.compare_digest(state, expected):
            return RedirectResponse(f"{PREFIX}/auth/denied?why=state", status_code=302)

        try:
            email = await auth.exchange_and_verify(cfg, code, nonce or "")
        except auth.AuthError as exc:
            return RedirectResponse(
                f"{PREFIX}/auth/denied?why={urllib.parse.quote(str(exc))}", status_code=302
            )

        response = RedirectResponse(PREFIX, status_code=302)
        response.set_cookie(
            auth.COOKIE,
            auth.issue_session(email, cfg.session_secret),
            max_age=auth.SESSION_TTL_S,
            httponly=True,
            secure=True,
            samesite="lax",
        )
        for name in ("lad_oauth_state", "lad_oauth_nonce"):
            response.delete_cookie(name)
        return response

    @app.get(f"{PREFIX}/auth/denied")
    async def denied(why: str = "Sign-in failed."):
        return Response(
            f"{why}\n\nTry again: {PREFIX}/auth/login\n",
            status_code=403,
            media_type="text/plain",
        )

    @app.post(f"{PREFIX}/auth/logout")
    async def logout():
        response = RedirectResponse(f"{PREFIX}/auth/login", status_code=302)
        response.delete_cookie(auth.COOKIE)
        return response

    @app.get(f"{PREFIX}/api/whoami")
    async def whoami(request: Request):
        return {"email": getattr(request.state, "email", None)}

    outputs.install(app, PREFIX)

    return app
