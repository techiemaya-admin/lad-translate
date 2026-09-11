"""
Operator API for hardware audio output.

Device profiles and channel maps, read and written by the portal in
LAD-Frontend. This service owns the data and the invariants; the portal owns
the interface and the per-user authorisation.

    GET     /api/admin/outputs/devices              every device for a tenant
    POST    /api/admin/outputs/devices              create one, id assigned here
    GET     /api/admin/outputs/devices/{device_id}  one device and its map
    PUT     /api/admin/outputs/devices/{device_id}  replace it and its whole map
    DELETE  /api/admin/outputs/devices/{device_id}  remove it

SEPARATE APP, SEPARATE PORT. This is deliberately not mounted on the listener
join service. That service is reachable by every phone in the room, and hanging
an operator API off the same origin means the only thing between the audience
and the venue's patch is a bearer token. Run it with tools/serve_admin.py, bound
where the audience network cannot reach it.

AUTHENTICATION IS SERVICE-TO-SERVICE. One shared bearer token, from
LAD_ADMIN_TOKEN, identifying the portal rather than a person. There is no
default: an unset token disables the API instead of opening it, the same rule
db/pool.py applies to the database URL, and for the same reason -- a
wrong-but-plausible default connects, works, and lets in whoever finds it.
Knowing WHICH operator changed a patch is the portal's job, and when that
matters here the token becomes a per-user credential and these handlers grow an
actor argument.

The tenant is explicit, in X-Tenant-Id, and resolved through TenantResolver so
an inactive or unknown tenant is an error rather than a schema guess.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import OutputChannel, OutputDevice, TenantContext
from ..db.outputs import DuplicateDeviceName, OutputStore
from ..db.pool import create_pool
from ..db.tenancy import SchemaError, TenantResolver
from ..obs.log import get_logger

log = get_logger(__name__)


# Module level, not nested in the factory. This module uses
# `from __future__ import annotations`, so FastAPI resolves handler annotations
# against module globals; a model defined inside create_admin_app is invisible
# there and the request body silently degrades to a query parameter. The same
# trap is documented in api/join.py.
class ChannelBody(BaseModel):
    language: str
    channel: int = Field(ge=1, le=64)
    ir_channel: int | None = Field(default=None, ge=1, le=99)
    label: str = ""
    gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)
    enabled: bool = True


class DeviceBody(BaseModel):
    name: str
    device_name: str
    channel_count: int = Field(ge=1, le=64)
    kind: str = "dante-vsc"
    sample_rate: int = 48000
    enabled: bool = True
    channels: list[ChannelBody] = Field(default_factory=list)


def _channel_json(channel: OutputChannel) -> dict:
    return {
        "language": channel.language,
        "channel": channel.channel,
        "ir_channel": channel.ir_channel,
        "label": channel.label,
        "gain_db": round(channel.gain_db, 2),
        "enabled": channel.enabled,
    }


def _device_json(device: OutputDevice) -> dict:
    return {
        "device_id": device.device_id,
        "name": device.name,
        "kind": device.kind,
        "device_name": device.device_name,
        "channel_count": device.channel_count,
        "sample_rate": device.sample_rate,
        "enabled": device.enabled,
        "languages": device.languages,
        "channels": [_channel_json(c) for c in sorted(device.channels, key=lambda c: c.channel)],
    }


def _to_device(body: DeviceBody, device_id: str) -> OutputDevice:
    """
    Build the domain object, letting its own validation speak.

    OutputDevice.__post_init__ enforces what Pydantic cannot see across fields:
    a channel beyond the device's channel count, two languages on one wire, two
    on one IR channel. Its messages say what is wrong in the operator's terms,
    so they are returned as-is rather than replaced with a generic 400.
    """
    try:
        return OutputDevice(
            device_id=device_id,
            name=body.name,
            kind=body.kind,
            device_name=body.device_name,
            channel_count=body.channel_count,
            sample_rate=body.sample_rate,
            enabled=body.enabled,
            channels=tuple(
                OutputChannel(
                    language=c.language,
                    channel=c.channel,
                    ir_channel=c.ir_channel,
                    label=c.label,
                    gain_db=c.gain_db,
                    enabled=c.enabled,
                )
                for c in body.channels
            ),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def store_for(
    request: Request,
    authorization: Annotated[str, Header()] = "",
    x_tenant_id: Annotated[str, Header()] = "",
) -> OutputStore:
    """
    Authenticate the caller, resolve the tenant, hand back its store.

    Module level, like the request models above, and for the same reason. This
    module uses `from __future__ import annotations`, so a handler's
    `Annotated[OutputStore, Depends(store_for)]` reaches FastAPI as a string
    resolved against module globals; defined inside create_admin_app this
    function is not there, the annotation fails to resolve, and `store`
    degrades into a required query parameter that every request then fails to
    supply. Configuration comes off request.app.state rather than a closure.
    """
    state = request.app.state
    if not state.admin_token:
        raise HTTPException(503, "operator API is not configured; set LAD_ADMIN_TOKEN")

    scheme, _, presented = authorization.partition(" ")
    # compare_digest, not ==: string equality returns early on the first
    # differing byte, and the timing of that is enough to recover a token one
    # byte at a time.
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented, state.admin_token):
        raise HTTPException(401, "bad or missing bearer token")

    if not x_tenant_id:
        raise HTTPException(400, "X-Tenant-Id header is required")
    if state.pool is None or state.resolver is None:
        raise HTTPException(503, "database not configured")

    try:
        tenant: TenantContext = await state.resolver.resolve(
            x_tenant_id, state.database_url or "unused"
        )
    except SchemaError as exc:
        # 404, not 403: whether a tenant id exists is not something an
        # authenticated portal should have to guess at, and this token already
        # speaks for the whole platform.
        raise HTTPException(404, str(exc)) from exc

    return OutputStore(state.pool, tenant)


def create_admin_app(
    pool=None,
    control_schema: str | None = None,
    database_url: str | None = None,
    admin_token: str | None = None,
):
    """
    Build the operator API.

    Pass `pool` to supply your own (tests), or `database_url` to have one opened
    in the lifespan -- asyncpg binds a pool to the loop that created it, so a
    pool built before uvicorn starts fails on first use.
    """
    from contextlib import asynccontextmanager

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
        title="LAD Live Translation — operator API",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.pool = pool
    app.state.control_schema = control_schema or os.getenv("LAD_CONTROL_SCHEMA")
    # Built here when the pool was handed in, and again in the lifespan when
    # this app opens its own. A caller that supplies a pool may also drive the
    # app without running the lifespan at all -- httpx's ASGITransport does
    # exactly that -- and leaving the resolver until then makes every request
    # answer "database not configured" against a perfectly good pool.
    app.state.resolver = (
        TenantResolver(pool, app.state.control_schema)
        if pool is not None and app.state.control_schema
        else None
    )
    app.state.admin_token = admin_token or os.getenv("LAD_ADMIN_TOKEN") or ""
    app.state.database_url = database_url or os.getenv("LAD_DATABASE_URL", "")

    if not app.state.admin_token:
        log.warning(
            "LAD_ADMIN_TOKEN is not set; the operator API will refuse every request"
        )

    # -------------------------------------------------------------------------

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "configured": bool(app.state.admin_token)}

    @app.get("/api/admin/outputs/devices")
    async def list_devices(store: Annotated[OutputStore, Depends(store_for)]):
        devices = await store.list_devices()
        return {"devices": [_device_json(d) for d in devices]}

    @app.post("/api/admin/outputs/devices", status_code=201)
    async def create_device(body: DeviceBody, store: Annotated[OutputStore, Depends(store_for)]):
        # Empty device_id means OutputStore assigns one.
        try:
            saved = await store.save_device(_to_device(body, ""))
        except DuplicateDeviceName as exc:
            raise HTTPException(409, str(exc)) from exc
        return _device_json(saved)

    @app.get("/api/admin/outputs/devices/{device_id}")
    async def get_device(device_id: str, store: Annotated[OutputStore, Depends(store_for)]):
        device = await store.get_device(device_id)
        if device is None:
            raise HTTPException(404, "no such device for this tenant")
        return _device_json(device)

    @app.put("/api/admin/outputs/devices/{device_id}")
    async def replace_device(
        device_id: str,
        body: DeviceBody,
        store: Annotated[OutputStore, Depends(store_for)],
    ):
        """
        Replace an existing device and its entire channel map.

        PUT rather than PATCH because the portal sends the patch the operator
        drew, not the moves they made to draw it. A dropped request then leaves
        the previous map intact instead of a half-repatched rig, and freeing a
        channel and reusing it in one edit does not collide with itself.

        It edits and does not create, so creation stays on POST. Every tenant
        has its own schema, so a PUT carrying an id this tenant does not own
        would otherwise write a perfectly valid device into their schema and
        answer 200 -- no leak, but a phantom rig conjured out of a stale id or
        a typo, which an operator then has to find and delete.
        """
        if await store.get_device(device_id) is None:
            raise HTTPException(404, "no such device for this tenant")
        try:
            saved = await store.save_device(_to_device(body, device_id))
        except DuplicateDeviceName as exc:
            raise HTTPException(409, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _device_json(saved)

    @app.delete("/api/admin/outputs/devices/{device_id}", status_code=204)
    async def delete_device(device_id: str, store: Annotated[OutputStore, Depends(store_for)]):
        if not await store.delete_device(device_id):
            raise HTTPException(404, "no such device for this tenant")

    return app
