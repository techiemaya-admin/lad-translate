"""
Tenant resolution.

Single database, one schema per tenant, tenant_id on every row. The same model
VOAG uses, with the part that breaks multi-tenancy left out.

VOAG resolves its schema once, at import:

    SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")          # schema_constants.py:18
    CALL_LOGS_FULL = f"{SCHEMA}.{CALL_LOGS_TABLE}"      # baked at import

Two consequences. One container can only ever serve one tenant, because the
constant is process-wide. And when nothing resolves, it silently writes into
the shared control plane rather than failing.

Here the schema is looked up per tenant and passed explicitly. There is no
module-level constant, no environment default, and no fallback.

On validation: a schema name is interpolated into SQL because identifiers
cannot be parameterised, so it is the one value that must be beyond doubt.
VOAG's sanitizeSchema() strips unexpected characters, which silently turns one
name into a different valid one. This rejects instead. A schema that does not
match the pattern is an error, never something quietly repaired.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..config import TenantContext
from ..obs.log import get_logger

log = get_logger(__name__)

# Lowercase, starts with a letter, then letters, digits and underscores.
# Postgres identifiers cap at 63 bytes.
SCHEMA_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

# Names that are legal identifiers but must never be a tenant's schema.
RESERVED_SCHEMAS = frozenset({"public", "information_schema", "pg_catalog", "pg_toast"})


class SchemaError(ValueError):
    """A schema name failed validation. Never repaired, always raised."""


def validate_schema(name: str) -> str:
    """
    Check a schema name is safe to interpolate, or raise.

    Returns the name unchanged on success. It is deliberately not a sanitiser:
    silently rewriting an unexpected name is how a query ends up pointed at the
    wrong tenant's data.
    """
    if not name:
        raise SchemaError("schema name is empty; there is no default schema")
    if not SCHEMA_PATTERN.match(name):
        raise SchemaError(
            f"schema name {name!r} is not a safe identifier; expected "
            f"{SCHEMA_PATTERN.pattern}"
        )
    if name in RESERVED_SCHEMAS:
        raise SchemaError(f"schema name {name!r} is reserved")
    return name


@dataclass(frozen=True, slots=True)
class TenantRecord:
    tenant_id: str
    slug: str
    schema: str
    is_active: bool


class TenantResolver:
    """
    Maps tenant_id to schema, through the control plane directory.

    Results are cached because a session resolves once at start and then holds
    the context for hours. The TTL exists so a tenant deactivated mid-event is
    not served indefinitely by a stale entry.
    """

    def __init__(self, pool, control_schema: str, ttl_s: float = 300.0) -> None:
        self._pool = pool
        self._control_schema = validate_schema(control_schema)
        self._ttl_s = ttl_s
        self._cache: dict[str, tuple[float, TenantRecord]] = {}

    @property
    def control_schema(self) -> str:
        return self._control_schema

    async def resolve(self, tenant_id: str, database_url: str) -> TenantContext:
        """
        Build the context a session carries for its whole life.

        Raises rather than falling back. An unresolved tenant means the caller
        has a bug or a bad request, and writing somewhere plausible instead is
        how data lands in the wrong place.
        """
        record = await self.lookup(tenant_id)
        if not record.is_active:
            raise SchemaError(f"tenant {tenant_id} is not active")
        return TenantContext(
            tenant_id=record.tenant_id,
            database_url=database_url,
            schema=record.schema,
        )

    async def resolve_slug(self, slug: str, database_url: str) -> TenantContext:
        """
        The same, keyed by slug.

        The SFU host is configured with a slug (LAD_TRANSLATE_TENANT=techiemaya)
        because a human types it into session.env; ids are what services pass
        each other. tools/serve_session.py did this lookup inline; the console
        needs it too, and a second inline copy is one more place for the
        `is_active` clause to go missing.
        """
        record = await self.lookup(slug, by="slug")
        if not record.is_active:
            raise SchemaError(f"tenant {slug} is not active")
        return TenantContext(
            tenant_id=record.tenant_id,
            database_url=database_url,
            schema=record.schema,
        )

    async def lookup(self, key: str, by: str = "id") -> TenantRecord:
        if by not in ("id", "slug"):
            raise ValueError(f"lookup by {by!r}; expected 'id' or 'slug'")
        cache_key = f"{by}:{key}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < self._ttl_s:
            return cached[1]

        # The control schema is validated at construction, so this
        # interpolation is safe; so is the WHERE clause, chosen from two
        # literals above. The key itself is parameterised.
        where = "id = $1::uuid" if by == "id" else "slug = $1"
        row = await self._pool.fetchrow(
            f"""
            SELECT id::text, slug, schema_name, is_active
            FROM {self._control_schema}.tenants
            WHERE {where}
            """,
            key,
        )
        if row is None:
            raise SchemaError(f"no tenant {key} in {self._control_schema}.tenants")
        tenant_id = row[0]

        record = TenantRecord(
            tenant_id=row[0],
            slug=row[1],
            # Validated on the way out as well as by the table constraint.
            # The check costs nothing and the failure mode it prevents is
            # querying another tenant's data.
            schema=validate_schema(row[2]),
            is_active=row[3],
        )
        self._cache[cache_key] = (time.monotonic(), record)
        log.info(
            "tenant resolved",
            extra={"tenant_id": tenant_id, "slug": record.slug, "schema": record.schema},
        )
        return record

    def invalidate(self, tenant_id: str | None = None) -> None:
        """Drop cached entries. Call when a tenant's configuration changes."""
        if tenant_id is None:
            self._cache.clear()
        else:
            self._cache.pop(f"id:{tenant_id}", None)
            # A tenant is one row however it was looked up.
            for key in [k for k, v in self._cache.items() if v[1].tenant_id == tenant_id]:
                self._cache.pop(key, None)
