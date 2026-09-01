#!/usr/bin/env python3
"""
The bits of the Postgres client tooling this project needs, over asyncpg.

`pg_isready`, `createdb` and `psql` are client programs, and not every Postgres
distribution ships them. The Windows build under `.local/pgsql` carries the
server -- `initdb`, `pg_ctl`, `postgres` -- and nothing else, so `pg.ps1` has no
binary to call for "is it up yet" or "make the database". asyncpg is already a
dependency of the project and speaks the same wire protocol, so the three things
actually needed are implemented against it rather than against a client package
that has to be installed separately.

    python tools/pg_admin.py ready --wait 20      exit 0 once it accepts connections
    python tools/pg_admin.py createdb             create the database if absent
    python tools/pg_admin.py sql "SELECT 1"       run a statement, print the rows

`ready` polls rather than asking once. A cluster that has just been started
refuses connections for a moment while it recovers, and a single probe against
that window reports a healthy server as down.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from urllib.parse import urlsplit, urlunsplit

DEFAULT_URL = "postgresql://lad@127.0.0.1:55432/salesmaya_agent"


def maintenance_url(url: str) -> str:
    """The same server, on the `postgres` database.

    Creating a database means being connected to a different one, and the target
    is by definition not there yet.
    """
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/postgres"))


def database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/") or "postgres"


async def wait_ready(url: str, seconds: float) -> int:
    import asyncpg

    deadline = asyncio.get_running_loop().time() + seconds
    last = ""
    while True:
        try:
            conn = await asyncpg.connect(maintenance_url(url), timeout=2)
            await conn.close()
            return 0
        except Exception as exc:
            # Any failure at all means not ready: connection refused while the
            # postmaster is still binding, and "the database system is starting
            # up" while it recovers, are both a server that is simply not there
            # yet, and neither is worth distinguishing from the other here.
            last = f"{type(exc).__name__}: {exc}"
        if asyncio.get_running_loop().time() >= deadline:
            print(f"not ready after {seconds:g}s: {last}", file=sys.stderr)
            return 1
        await asyncio.sleep(0.5)


async def create_database(url: str) -> int:
    import asyncpg

    name = database_name(url)
    conn = await asyncpg.connect(maintenance_url(url))
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if exists:
            print(f"database {name} already exists")
            return 0
        # Identifiers cannot be parameterised. The name comes from the URL this
        # tooling owns, and is quoted rather than interpolated bare.
        await conn.execute(f'CREATE DATABASE "{name}"')
        print(f"database {name} created")
        return 0
    finally:
        await conn.close()


async def run_sql(url: str, statement: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(url)
    try:
        try:
            rows = await conn.fetch(statement)
        except asyncpg.exceptions.PostgresSyntaxError:
            raise
        for row in rows:
            print("  ".join("" if v is None else str(v) for v in row.values()))
        if not rows:
            print("(no rows)")
        return 0
    finally:
        await conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="database URL (default: LAD_DATABASE_URL)")
    sub = ap.add_subparsers(dest="command", required=True)

    ready = sub.add_parser("ready", help="exit 0 once the server accepts connections")
    ready.add_argument("--wait", type=float, default=20.0, help="seconds to keep trying")

    sub.add_parser("createdb", help="create the database in the URL if it is absent")

    sql = sub.add_parser("sql", help="run one statement and print the rows")
    sql.add_argument("statement")

    args = ap.parse_args()
    url = args.url or os.getenv("LAD_DATABASE_URL") or DEFAULT_URL

    if args.command == "ready":
        return asyncio.run(wait_ready(url, args.wait))
    if args.command == "createdb":
        return asyncio.run(create_database(url))
    return asyncio.run(run_sql(url, args.statement))


if __name__ == "__main__":
    raise SystemExit(main())
