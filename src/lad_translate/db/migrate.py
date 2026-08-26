"""
Migration runner.

Migrations are SQL with a {schema} placeholder, applied once per target schema.
The control plane migration goes into the control schema; the rest go into each
tenant's schema.

Deliberately simple: no version table beyond applied_migrations, no down
migrations, no branching. Session tables are additive and a live event is not
the place to discover a clever migration framework.
"""

from __future__ import annotations

from pathlib import Path

from ..obs.log import get_logger
from .tenancy import validate_schema

log = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

CONTROL_MIGRATIONS = ("000_control_plane.sql",)
TENANT_MIGRATIONS = ("001_translation_sessions.sql",)

_LEDGER = """
CREATE TABLE IF NOT EXISTS {schema}.applied_migrations (
    filename    text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


async def apply(pool, schema: str, filenames: tuple[str, ...]) -> list[str]:
    """
    Apply any of `filenames` not yet recorded for `schema`.

    Returns the ones applied. Safe to call on every start-up.
    """
    schema = validate_schema(schema)
    applied: list[str] = []

    async with pool.acquire() as conn:
        # The first migration creates the schema, so the ledger has to wait
        # until after it on a fresh schema.
        for filename in filenames:
            sql = (MIGRATIONS_DIR / filename).read_text().replace("{schema}", schema)
            async with conn.transaction():
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                await conn.execute(_LEDGER.format(schema=schema))
                already = await conn.fetchval(
                    f"SELECT 1 FROM {schema}.applied_migrations WHERE filename = $1",
                    filename,
                )
                if already:
                    continue
                await conn.execute(sql)
                await conn.execute(
                    f"INSERT INTO {schema}.applied_migrations (filename) VALUES ($1)",
                    filename,
                )
                applied.append(filename)
                log.info("migration applied", extra={"schema": schema, "migration": filename})
    return applied


async def apply_control(pool, schema: str) -> list[str]:
    return await apply(pool, schema, CONTROL_MIGRATIONS)


async def apply_tenant(pool, schema: str) -> list[str]:
    return await apply(pool, schema, TENANT_MIGRATIONS)
