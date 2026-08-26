"""
Connection pooling.

One database, one pool. Which schema a query targets is decided per call from
the TenantContext, never by the connection's search_path: a pooled connection
is shared, so setting search_path on it leaks one tenant's schema into the next
tenant's query.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from ..obs.log import get_logger

log = get_logger(__name__)


def database_url() -> str:
    """
    The database URL, from the environment. No default.

    A wrong-but-plausible default is worse than a missing one: it connects, it
    works, and it writes to somewhere nobody meant.
    """
    url = os.getenv("LAD_DATABASE_URL")
    if not url:
        raise RuntimeError("LAD_DATABASE_URL is not set; see .env.example")
    return url


def control_schema() -> str:
    """Schema holding the tenant directory. From the environment, no default."""
    schema = os.getenv("LAD_CONTROL_SCHEMA")
    if not schema:
        raise RuntimeError("LAD_CONTROL_SCHEMA is not set; see .env.example")
    return schema


async def create_pool(url: str | None = None, min_size: int = 2, max_size: int = 10):
    import asyncpg

    resolved = url or database_url()
    pool = await asyncpg.create_pool(resolved, min_size=min_size, max_size=max_size)
    log.info("database pool opened", extra={"min_size": min_size, "max_size": max_size})
    return pool


@asynccontextmanager
async def pool_context(url: str | None = None, **kwargs):
    pool = await create_pool(url, **kwargs)
    try:
        yield pool
    finally:
        await pool.close()
        log.info("database pool closed")
