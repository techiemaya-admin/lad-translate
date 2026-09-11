"""
Hardware output in the operator console: the venue's channel map.

The five operations live in api/admin.py and are shared with the portal API.
What differs here is the door. The portal API is service-to-service - one
bearer token for the whole platform and an explicit X-Tenant-Id - because a
portal speaks for many tenants. The console is a person, already signed in
with Google, on the box that serves exactly one tenant. So there is no token
to present and no tenant to name: the tenant is the one in session.env, the
same one the session process resolves, and the operator's identity is the
cookie the middleware already checked.

The brief that specified the portal API put the editor in LAD-Frontend. It is
here instead because this is where the venue operator already is: the person
patching French onto IR channel 2 at eight in the morning is the same person
who picked the preset and printed the QR codes, and a second portal for the
second job means two sign-ins and two tabs on a laptop balanced on a flight
case. The portal API stays, unchanged, for a portal that wants it.

Three states this must tell apart, because to the page they would otherwise
all look like "no devices":

  - the console has no database configured           -> not this box's job yet
  - the tenant's schema has not had migration 002    -> run seed_tenant.py
  - there are no devices                             -> add one

The first two are reported as such rather than as an empty list. An empty
list is an answer; these are the absence of one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response

from ..api.admin import (
    DeviceBody,
    device_create,
    device_delete,
    device_get,
    device_replace,
    devices_list,
)
from ..api.languages import describe
from ..config import DEVICE_KINDS, DEVICE_SAMPLE_RATES
from ..db.outputs import OutputStore
from ..db.tenancy import SchemaError, TenantResolver
from ..obs.log import get_logger
from . import env

log = get_logger(__name__)

KINDS_WITH_AN_ENGINE: dict[str, str] = {
    "aes67": "session/aes67.py, run by tools/output_agent.py at the venue",
}
"""
Which device kinds actually route audio, and by what.

The dropdown offers every value the schema accepts, because a profile saved
for a rig that arrives next month is a reasonable thing to store. But a value
in a dropdown reads as a capability, and four of the five are not one. The
page labels each option from this table so the dropdown says so itself. Add a
kind here when its sink lands, not before.
"""

UNDEFINED_TABLE = "42P01"
"""Postgres: relation does not exist. What a tenant schema without migration
002 answers, and the one database error that is a configuration state rather
than a bug."""


@dataclass
class OutputsConfig:
    """What the console needs to reach one tenant's channel maps."""

    pool: object | None
    resolver: TenantResolver | None
    tenant_slug: str
    database_url: str

    @property
    def configured(self) -> bool:
        return self.pool is not None and self.resolver is not None and bool(self.tenant_slug)

    async def tenant(self):
        """
        Resolve the box's tenant. Raises SchemaError if it is not seeded.

        TenantContext insists on a database_url because a session carries it
        for its whole life; here the pool is what does the work and the URL
        is informational, and a pool handed in by a test has no URL at all.
        Same accommodation api/admin.py makes.
        """
        return await self.resolver.resolve_slug(self.tenant_slug, self.database_url or "unused")

    @property
    def why_not(self) -> str:
        if self.pool is None or self.resolver is None:
            return (
                "The console has no database. Set LAD_DATABASE_URL and "
                "LAD_CONTROL_SCHEMA for the console service; on the VM, "
                "re-run deploy/vm/bootstrap.sh."
            )
        if not self.tenant_slug:
            return "LAD_TRANSLATE_TENANT is not set, so there is no tenant to hold a channel map."
        return ""


def _migration_hint(slug: str) -> str:
    return (
        "This tenant's schema does not have the hardware output tables yet. "
        f"Run: tools/seed_tenant.py --slug {slug} --migrate-only"
    )


async def store_for(request: Request) -> OutputStore:
    """
    The tenant's store, or a 503 that says which of the three states we are in.

    503 and not 500: none of these is a fault in the request, and none is
    something the operator can fix from the page. It is the box that is not
    ready, and the page should say so in those words.
    """
    cfg: OutputsConfig = request.app.state.outputs
    if not cfg.configured:
        raise HTTPException(503, cfg.why_not)
    try:
        tenant = await cfg.tenant()
    except SchemaError as exc:
        raise HTTPException(
            503, f"{exc}. Seed it with: tools/seed_tenant.py --slug {cfg.tenant_slug}"
        ) from exc
    return OutputStore(cfg.pool, tenant)


async def _guarded(coro, slug: str):
    """
    Run one operation, turning a missing table into the migration hint.

    Matched on sqlstate rather than by importing asyncpg's exception type, the
    same way db/outputs.py identifies a constraint clash: this module should
    not have to know which driver is underneath to say "run the migration".
    """
    try:
        return await coro
    except Exception as exc:
        if getattr(exc, "sqlstate", "") == UNDEFINED_TABLE:
            raise HTTPException(503, _migration_hint(slug)) from exc
        raise


def _languages(env_path: Path) -> list[dict]:
    """
    The languages this box actually publishes, source first.

    From session.env rather than from a global list: a channel mapped to a
    language the session does not produce is a wire that will carry silence
    all day, and the dropdown should not offer it. The source language is
    included because venues put the floor feed on a channel for the booth
    and the recorder.
    """
    current = env.read(env_path)
    targets = [t.strip() for t in current.get("LAD_TRANSLATE_TARGETS", "").split(",") if t.strip()]
    source = current.get("LAD_TRANSLATE_SOURCE", "en").strip() or "en"
    codes = [source] + [t for t in targets if t != source]
    out = []
    for code in codes:
        info = describe(code)
        out.append({"code": code, "native": info.native, "english": info.english})
    return out


def signage_text(device: dict, languages: list[dict]) -> str:
    """
    What goes on the wall next to the handset table.

    IR channel numbers, in order, with the language in its own script and in
    English. Only channels with an IR number and only enabled ones: a channel
    feeding the recorder is not something an audience member can select.
    """
    names = {lang["code"]: lang for lang in languages}
    rows = sorted(
        (c for c in device["channels"] if c["ir_channel"] is not None and c["enabled"]),
        key=lambda c: c["ir_channel"],
    )
    lines = [f"{device['name']} — IR channels", ""]
    if not rows:
        lines.append("No IR channels are assigned on this device.")
    for c in rows:
        lang = names.get(c["language"])
        native = lang["native"] if lang else describe(c["language"]).native
        english = lang["english"] if lang else describe(c["language"]).english
        label = f"  ({english})" if english != native else ""
        lines.append(f"  {c['ir_channel']:>2}   {native}{label}")
    return "\n".join(lines) + "\n"


def install(app: FastAPI, prefix: str) -> None:
    """Mount the routes on the console app. The middleware gates them: the
    OPEN allowlist in console/app.py does not include any of these paths."""

    base = f"{prefix}/api/outputs"

    @app.get(base)
    async def overview(request: Request):
        """
        Everything the page needs in one fetch, including WHY there is nothing.

        200 in every state. The page renders the reason as a state, not as an
        error toast, because "the box has no database" is information the
        operator needs to read once, not a failure to retry.
        """
        cfg: OutputsConfig = request.app.state.outputs
        languages = _languages(request.app.state.env_path)
        shell = {
            "configured": cfg.configured,
            "migrated": None,
            "reason": cfg.why_not,
            "tenant": None,
            "devices": [],
            "languages": languages,
            "kinds": [
                {"key": kind, "built": kind in KINDS_WITH_AN_ENGINE,
                 "engine": KINDS_WITH_AN_ENGINE.get(kind)}
                for kind in DEVICE_KINDS
            ],
            "sample_rates": list(DEVICE_SAMPLE_RATES),
        }
        if not cfg.configured:
            return shell

        try:
            tenant = await cfg.tenant()
        except SchemaError as exc:
            shell["reason"] = f"{exc}. Seed it with: tools/seed_tenant.py --slug {cfg.tenant_slug}"
            return shell
        shell["tenant"] = {"slug": cfg.tenant_slug, "schema": tenant.schema}

        store = OutputStore(cfg.pool, tenant)
        try:
            listed = await devices_list(store)
        except Exception as exc:
            if getattr(exc, "sqlstate", "") == UNDEFINED_TABLE:
                shell["migrated"] = False
                shell["reason"] = _migration_hint(cfg.tenant_slug)
                return shell
            raise
        shell["migrated"] = True
        shell["devices"] = listed["devices"]
        return shell

    @app.post(f"{base}/devices", status_code=201)
    async def create(body: DeviceBody, request: Request):
        store = await store_for(request)
        saved = await _guarded(device_create(store, body), request.app.state.outputs.tenant_slug)
        log.info(
            "console saved output device",
            extra={
                "operator": getattr(request.state, "email", None),
                "device_id": saved["device_id"],
                "device": saved["name"],
                "channels": len(saved["channels"]),
                "is_new": True,
            },
        )
        return saved

    @app.get(f"{base}/devices/{{device_id}}")
    async def get(device_id: str, request: Request):
        store = await store_for(request)
        return await _guarded(device_get(store, device_id), request.app.state.outputs.tenant_slug)

    @app.put(f"{base}/devices/{{device_id}}")
    async def replace(device_id: str, body: DeviceBody, request: Request):
        store = await store_for(request)
        saved = await _guarded(
            device_replace(store, device_id, body), request.app.state.outputs.tenant_slug
        )
        log.info(
            "console saved output device",
            extra={
                "operator": getattr(request.state, "email", None),
                "device_id": device_id,
                "device": saved["name"],
                "channels": len(saved["channels"]),
                "is_new": False,
            },
        )
        return saved

    @app.delete(f"{base}/devices/{{device_id}}", status_code=204)
    async def delete(device_id: str, request: Request):
        store = await store_for(request)
        await _guarded(device_delete(store, device_id), request.app.state.outputs.tenant_slug)
        log.info(
            "console deleted output device",
            extra={"operator": getattr(request.state, "email", None), "device_id": device_id},
        )
        return Response(status_code=204)

    @app.get(f"{base}/devices/{{device_id}}/signage")
    async def signage(device_id: str, request: Request):
        store = await store_for(request)
        device = await _guarded(device_get(store, device_id), request.app.state.outputs.tenant_slug)
        text = signage_text(device, _languages(request.app.state.env_path))
        return Response(
            content=text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'inline; filename="signage-{device_id[:8]}.txt"'},
        )
