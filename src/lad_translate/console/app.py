"""
Operator console.

Served from the VM behind Caddy's basic auth, NOT from the Cloud Run join
service. That service is deliberately --allow-unauthenticated, because a
listener scans a QR code and has no credentials; a surface that can restart
sessions and change models cannot share that door.

It also has to reach systemd, which is local to this box and unreachable from
Cloud Run.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..obs.log import get_logger
from . import env, sessions
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


def create_app(public_base: str, env_path: Path | None = None) -> FastAPI:
    """
    `public_base` is the URL a PHONE reaches the join service on - the Cloud Run
    address, not this box. The console runs on the SFU host; the QR codes it
    prints must not point at it.
    """
    app = FastAPI(title="LAD Live Translation - console", docs_url=None, redoc_url=None)
    app.state.public_base = public_base.rstrip("/")
    app.state.env_path = env_path or env.DEFAULT_PATH

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

    return app
