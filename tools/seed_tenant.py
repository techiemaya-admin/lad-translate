#!/usr/bin/env python3
"""
Create a tenant in the local database and migrate its schema.

Applies the control plane migration if needed, registers the tenant, then
applies the session tables into that tenant's own schema.

Usage:
    python tools/seed_tenant.py --slug techiemaya
    python tools/seed_tenant.py --slug techiemaya --schema tenant_techiemaya
    python tools/seed_tenant.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lad_translate.db import migrate
from lad_translate.db.tenancy import validate_schema
from lad_translate.obs.log import configure

DEFAULT_URL = "postgresql://lad@127.0.0.1:55432/salesmaya_agent"


def schema_from_slug(slug: str) -> str:
    """Derive a schema name, then validate it. Never silently repaired."""
    candidate = "tenant_" + re.sub(r"[^a-z0-9_]", "_", slug.lower())
    return validate_schema(candidate[:63])


async def run(args) -> int:
    import asyncpg

    url = args.url or os.getenv("LAD_DATABASE_URL") or DEFAULT_URL
    control = args.control or os.getenv("LAD_CONTROL_SCHEMA") or "lad_dev"

    pool = await asyncpg.create_pool(url, min_size=1, max_size=2)
    try:
        await migrate.apply_control(pool, control)

        if args.list:
            rows = await pool.fetch(
                f"SELECT id::text, slug, schema_name, is_active FROM {control}.tenants ORDER BY slug"
            )
            if not rows:
                print("no tenants")
            for r in rows:
                state = "active" if r[3] else "inactive"
                print(f"  {r[1]:<24} {r[2]:<28} {state}  {r[0]}")
            return 0

        schema = validate_schema(args.schema) if args.schema else schema_from_slug(args.slug)
        existing = await pool.fetchrow(
            f"SELECT id::text, schema_name FROM {control}.tenants WHERE slug = $1", args.slug
        )
        if existing:
            print(f"tenant {args.slug} already exists: {existing[0]} -> {existing[1]}")
            await migrate.apply_tenant(pool, existing[1])
            return 0

        tenant_id = str(uuid.uuid4())
        await pool.execute(
            f"""INSERT INTO {control}.tenants (id, slug, schema_name, display_name)
                VALUES ($1::uuid, $2, $3, $4)""",
            tenant_id,
            args.slug,
            schema,
            args.display_name or args.slug,
        )
        await migrate.apply_tenant(pool, schema)
        print(f"tenant   {args.slug}")
        print(f"id       {tenant_id}")
        print(f"schema   {schema}")
        return 0
    finally:
        await pool.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug")
    ap.add_argument("--schema", help="override the derived schema name")
    ap.add_argument("--display-name")
    ap.add_argument("--url", help="database URL (default: LAD_DATABASE_URL)")
    ap.add_argument("--control", help="control schema (default: LAD_CONTROL_SCHEMA)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if not args.list and not args.slug:
        ap.error("give --slug, or --list")
    configure("WARNING")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
