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

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/")
    async def page():
        return FileResponse(STATIC_DIR / "console.html")

    @app.get("/api/presets")
    async def presets():
        return {"presets": [p.as_dict() for p in PRESETS]}

    @app.get("/api/settings")
    async def settings():
        current = env.read(app.state.env_path)
        return {
            "settings": {k: v for k, v in current.items() if k in env.EDITABLE},
            "editable": sorted(env.EDITABLE),
            "public_base": app.state.public_base,
        }

    @app.get("/api/status")
    async def status(room: str = "dubai-demo"):
        try:
            return (await sessions.status(room)).__dict__
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/apply")
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

    @app.post("/api/stop")
    async def stop(body: ApplyRequest):
        try:
            await sessions.stop(body.room)
        except sessions.BadRoom as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"stopped": True}

    @app.get("/api/qr")
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

        import qrcode

        suffix = "/speak" if kind == "speak" else ""
        url = f"{app.state.public_base}/room/{room}{suffix}"

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
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"X-Encoded-Url": url, "Cache-Control": "no-store"},
        )

    return app
