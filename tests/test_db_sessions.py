"""
Session storage against a real Postgres.

Skipped when LAD_TEST_DATABASE_URL is unset. Start the local cluster with
tools/pg.sh start and point the variable at it.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lad_translate.config import (
    BackendSelection,
    LanguageTarget,
    SessionConfig,
    TenantContext,
)
from lad_translate.db import migrate
from lad_translate.db.sessions import SessionStore, TranscriptRow
from lad_translate.db.tenancy import SchemaError, TenantResolver

URL = os.getenv(
    "LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent"
)
CONTROL = "lad_test_control"

pytestmark = pytest.mark.asyncio


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(URL, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pool():
    if not await _reachable():
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")
    import asyncpg

    p = await asyncpg.create_pool(URL, min_size=1, max_size=4)
    await migrate.apply_control(p, CONTROL)
    yield p
    await p.close()


@pytest.fixture
async def tenants(pool):
    """Two tenants in separate schemas, so isolation can be tested for real."""
    made = []
    for label in ("a", "b"):
        tenant_id = str(uuid.uuid4())
        schema = f"lad_test_{label}_{tenant_id.replace('-', '')[:8]}"
        await pool.execute(
            f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid, $2, $3)",
            tenant_id,
            f"tenant-{label}-{tenant_id[:8]}",
            schema,
        )
        await migrate.apply_tenant(pool, schema)
        made.append(TenantContext(tenant_id=tenant_id, database_url=URL, schema=schema))
    yield made
    for ctx in made:
        await pool.execute(f"DROP SCHEMA IF EXISTS {ctx.schema} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", ctx.tenant_id)


def config_for(tenant: TenantContext, **over) -> SessionConfig:
    return SessionConfig(
        session_id=over.pop("session_id", str(uuid.uuid4())),
        tenant=tenant,
        room_name="room-1",
        event_name="Sharjah Summit",
        source_language="en",
        targets=[LanguageTarget("fr", "fr_FR-siwis-medium"), LanguageTarget("ar", "ar_JO-kareem-medium")],
        backends=BackendSelection(stt="faster-whisper", mt="opus-mt", tts="piper"),
        **over,
    )


# --- migrations -------------------------------------------------------------


async def test_migrations_are_idempotent(pool, tenants):
    """Runs on every start-up, so a second run must be a no-op."""
    schema = tenants[0].schema
    assert await migrate.apply_tenant(pool, schema) == []


async def test_migration_refuses_an_unsafe_schema_name(pool):
    with pytest.raises(SchemaError):
        await migrate.apply_tenant(pool, "bad-schema; drop table x")


# --- session lifecycle ------------------------------------------------------


async def test_session_lifecycle_and_billing(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])

    await store.create_session(cfg, latency_credible=False)
    await store.mark_live(cfg.session_id)

    row = await store.get_session(cfg.session_id)
    assert row["status"] == "live"
    assert row["latency_credible"] is False
    assert list(row["target_languages"]) == ["fr", "ar"]

    billing = await store.end_session(cfg.session_id)
    assert billing.language_count == 2, "billed on language count, not on calls"
    assert billing.seconds >= 0
    assert billing.language_seconds == billing.seconds * 2


async def test_billing_is_settled_at_close_not_recomputed(pool, tenants):
    """A later config change must not alter an invoice already raised."""
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)
    billing = await store.end_session(cfg.session_id)

    row = await store.get_session(cfg.session_id)
    assert row["billed_language_count"] == billing.language_count
    assert row["billed_seconds"] == billing.seconds


async def test_failed_session_records_why(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)
    await store.end_session(cfg.session_id, failure_reason="publisher lost ethernet")

    row = await store.get_session(cfg.session_id)
    assert row["status"] == "failed"
    assert row["failure_reason"] == "publisher lost ethernet"


async def test_ending_twice_is_refused(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)
    await store.end_session(cfg.session_id)
    with pytest.raises(LookupError):
        await store.end_session(cfg.session_id)


# --- tenant isolation -------------------------------------------------------


async def test_a_tenant_cannot_read_another_tenants_session(pool, tenants):
    """
    The property the whole tenancy design exists for.

    session_id alone would find this row. The tenant_id filter is what turns a
    wrong-tenant bug into an empty result instead of a data leak.
    """
    tenant_a, tenant_b = tenants
    store_a = SessionStore(pool, tenant_a)
    cfg = config_for(tenant_a)
    await store_a.create_session(cfg, latency_credible=False)

    store_b = SessionStore(pool, tenant_b)
    assert await store_b.get_session(cfg.session_id) is None


async def test_a_tenant_cannot_end_another_tenants_session(pool, tenants):
    tenant_a, tenant_b = tenants
    cfg = config_for(tenant_a)
    await SessionStore(pool, tenant_a).create_session(cfg, latency_credible=False)
    with pytest.raises(LookupError):
        await SessionStore(pool, tenant_b).end_session(cfg.session_id)


async def test_store_rejects_a_session_belonging_to_another_tenant(pool, tenants):
    tenant_a, tenant_b = tenants
    with pytest.raises(ValueError, match="does not match store tenant"):
        await SessionStore(pool, tenant_b).create_session(
            config_for(tenant_a), latency_credible=False
        )


async def test_live_sessions_are_scoped_to_the_tenant(pool, tenants):
    tenant_a, tenant_b = tenants
    await SessionStore(pool, tenant_a).create_session(config_for(tenant_a), latency_credible=False)
    assert len(await SessionStore(pool, tenant_a).live_sessions()) == 1
    assert await SessionStore(pool, tenant_b).live_sessions() == []


# --- listeners --------------------------------------------------------------


async def test_listener_counts_are_per_language(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)

    for _ in range(3):
        await store.listener_joined(cfg.session_id, "fr")
    leaving = await store.listener_joined(cfg.session_id, "ar")

    assert await store.active_listeners(cfg.session_id) == {"fr": 3, "ar": 1}
    await store.listener_left(leaving)
    assert await store.active_listeners(cfg.session_id) == {"fr": 3}


async def test_listener_leave_is_idempotent(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)
    listener = await store.listener_joined(cfg.session_id, "fr")
    await store.listener_left(listener)
    await store.listener_left(listener)
    assert await store.active_listeners(cfg.session_id) == {}


# --- transcripts ------------------------------------------------------------


def transcript_row(chunk_id=0, **over) -> TranscriptRow:
    return TranscriptRow(
        **{
            "chunk_id": chunk_id,
            "language": "fr",
            "source_text": "Good morning everyone and welcome,",
            "translated_text": "Bonjour a tous et bienvenue,",
            "t_audio_start": 0.0,
            "t_audio_end": 1.8,
            "latency_s": 1.24,
            "commit_reason": "clause",
            "revised": False,
            **over,
        }
    )


async def test_transcripts_are_stored_and_ordered(pool, tenants):
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)

    for i in (2, 0, 1):
        await store.record_transcript(cfg.session_id, transcript_row(chunk_id=i))

    rows = await store.transcript(cfg.session_id, "fr")
    assert [r["chunk_id"] for r in rows] == [0, 1, 2]
    assert rows[0]["latency_s"] == pytest.approx(1.24)
    assert rows[0]["commit_reason"] == "clause"


async def test_republishing_a_chunk_updates_rather_than_failing(pool, tenants):
    """A retried publish must not kill the session on a unique violation."""
    store = SessionStore(pool, tenants[0])
    cfg = config_for(tenants[0])
    await store.create_session(cfg, latency_credible=False)

    await store.record_transcript(cfg.session_id, transcript_row(translated_text="first"))
    await store.record_transcript(
        cfg.session_id, transcript_row(translated_text="corrected", revised=True)
    )

    rows = await store.transcript(cfg.session_id, "fr")
    assert len(rows) == 1
    assert rows[0]["translated_text"] == "corrected"
    assert rows[0]["revised"] is True


async def test_transcripts_are_isolated_between_tenants(pool, tenants):
    tenant_a, tenant_b = tenants
    store_a = SessionStore(pool, tenant_a)
    cfg = config_for(tenant_a)
    await store_a.create_session(cfg, latency_credible=False)
    await store_a.record_transcript(cfg.session_id, transcript_row())

    assert await SessionStore(pool, tenant_b).transcript(cfg.session_id, "fr") == []


# --- resolver ---------------------------------------------------------------


async def test_resolver_maps_tenant_to_schema(pool, tenants):
    resolver = TenantResolver(pool, CONTROL)
    ctx = await resolver.resolve(tenants[0].tenant_id, URL)
    assert ctx.schema == tenants[0].schema
    assert ctx.tenant_id == tenants[0].tenant_id


async def test_resolver_refuses_an_unknown_tenant(pool):
    resolver = TenantResolver(pool, CONTROL)
    with pytest.raises(SchemaError, match="no tenant"):
        await resolver.resolve(str(uuid.uuid4()), URL)


async def test_resolver_caches_then_can_be_invalidated(pool, tenants):
    resolver = TenantResolver(pool, CONTROL)
    first = await resolver.lookup(tenants[0].tenant_id)
    assert (await resolver.lookup(tenants[0].tenant_id)) is first
    resolver.invalidate(tenants[0].tenant_id)
    assert (await resolver.lookup(tenants[0].tenant_id)) is not first
