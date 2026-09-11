"""
Output profile storage against a real Postgres.

Skipped when LAD_TEST_DATABASE_URL is unset. Start the local cluster with
tools/pg.sh start and point the variable at it.

Two tenants throughout, for the same reason as test_db_sessions.py: tenant
isolation is the property worth a real database rather than a mock, because it
only means anything against real SQL.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lad_translate.config import OutputChannel, OutputDevice, TenantContext
from lad_translate.db import migrate
from lad_translate.db.outputs import DuplicateDeviceName, OutputStore

URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
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
    made = []
    for label in ("a", "b"):
        tenant_id = str(uuid.uuid4())
        schema = f"lad_out_{label}_{tenant_id.replace('-', '')[:8]}"
        await pool.execute(
            f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid, $2, $3)",
            tenant_id,
            f"out-{label}-{tenant_id[:8]}",
            schema,
        )
        await migrate.apply_tenant(pool, schema)
        made.append(TenantContext(tenant_id=tenant_id, database_url=URL, schema=schema))
    yield made
    for ctx in made:
        await pool.execute(f"DROP SCHEMA IF EXISTS {ctx.schema} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", ctx.tenant_id)


def a_device(**over) -> OutputDevice:
    base = dict(
        device_id="",
        name="Main hall DVS",
        device_name="Dante Virtual Soundcard",
        channel_count=16,
        channels=(
            OutputChannel(language="en", channel=1, label="Floor"),
            OutputChannel(language="fr", channel=2, ir_channel=1, label="Français"),
            OutputChannel(language="ar", channel=3, ir_channel=2, label="العربية"),
        ),
    )
    base.update(over)
    return OutputDevice(**base)


# --- round trip -------------------------------------------------------------


async def test_save_assigns_an_id_and_reads_back_whole(pool, tenants):
    store = OutputStore(pool, tenants[0])
    saved = await store.save_device(a_device())

    assert saved.device_id
    assert saved.channel_count == 16
    assert saved.languages == ["en", "fr", "ar"]
    assert [c.ir_channel for c in saved.channels] == [None, 1, 2]
    assert saved.channels_for("fr")[0].label == "Français"


async def test_listing_carries_every_device_with_its_map(pool, tenants):
    store = OutputStore(pool, tenants[0])
    await store.save_device(a_device(name="Main hall DVS"))
    await store.save_device(a_device(name="Breakout DVS", channel_count=8))

    devices = await store.list_devices()

    assert [d.name for d in devices] == ["Breakout DVS", "Main hall DVS"]
    assert all(len(d.channels) == 3 for d in devices)


async def test_a_replaced_map_does_not_leave_old_channels_behind(pool, tenants):
    store = OutputStore(pool, tenants[0])
    saved = await store.save_device(a_device())

    moved = OutputDevice(
        device_id=saved.device_id,
        name=saved.name,
        device_name=saved.device_name,
        channel_count=saved.channel_count,
        channels=(OutputChannel(language="hi", channel=1, ir_channel=3),),
    )
    again = await store.save_device(moved)

    assert again.languages == ["hi"]
    assert len(again.channels) == 1


async def test_a_channel_freed_and_reused_in_one_edit_is_allowed(pool, tenants):
    """
    Swapping two languages is one operator gesture. Row-by-row upserts would
    reject it on the unique constraint halfway through.
    """
    store = OutputStore(pool, tenants[0])
    saved = await store.save_device(a_device())

    swapped = OutputDevice(
        device_id=saved.device_id,
        name=saved.name,
        device_name=saved.device_name,
        channel_count=saved.channel_count,
        channels=(
            OutputChannel(language="ar", channel=2, ir_channel=2),
            OutputChannel(language="fr", channel=3, ir_channel=1),
        ),
    )
    again = await store.save_device(swapped)

    assert again.channels_for("ar")[0].channel == 2
    assert again.channels_for("fr")[0].channel == 3


async def test_delete_removes_the_device_and_its_channels(pool, tenants):
    store = OutputStore(pool, tenants[0])
    saved = await store.save_device(a_device())

    assert await store.delete_device(saved.device_id) is True
    assert await store.get_device(saved.device_id) is None

    remaining = await pool.fetchval(
        f"SELECT count(*) FROM {tenants[0].schema}.audio_output_channels"
    )
    assert remaining == 0


async def test_deleting_a_missing_device_reports_it_rather_than_raising(pool, tenants):
    store = OutputStore(pool, tenants[0])
    assert await store.delete_device(str(uuid.uuid4())) is False


# --- tenant isolation -------------------------------------------------------


async def test_one_tenant_cannot_read_anothers_device(pool, tenants):
    """A device_id alone identifies a device, so the tenant filter is the guard."""
    mine = OutputStore(pool, tenants[0])
    theirs = OutputStore(pool, tenants[1])
    saved = await mine.save_device(a_device())

    assert await theirs.get_device(saved.device_id) is None
    assert await theirs.list_devices() == []


async def test_one_tenant_cannot_delete_anothers_device(pool, tenants):
    mine = OutputStore(pool, tenants[0])
    theirs = OutputStore(pool, tenants[1])
    saved = await mine.save_device(a_device())

    assert await theirs.delete_device(saved.device_id) is False
    assert await mine.get_device(saved.device_id) is not None


async def test_writing_a_known_id_from_another_schema_leaves_the_original_alone(pool, tenants):
    """
    The dangerous case: a known device_id plus the wrong tenant.

    With a schema per tenant the write lands in the caller's own schema, so it
    creates their device rather than touching anyone else's. The original must
    come back untouched -- that is the property, not an error.
    """
    mine = OutputStore(pool, tenants[0])
    theirs = OutputStore(pool, tenants[1])
    saved = await mine.save_device(a_device())

    await theirs.save_device(a_device(device_id=saved.device_id, name="Hijack"))

    still = await mine.get_device(saved.device_id)
    assert still is not None
    assert still.name == "Main hall DVS"
    assert still.languages == ["en", "fr", "ar"]


async def test_a_shared_schema_still_separates_tenants_by_id(pool, tenants):
    """
    Defence in depth, and the reason every row carries tenant_id.

    Schema per tenant is a convention in the control plane, not something the
    data model can assume: nothing stops two tenants being pointed at one
    schema. There the schema name protects nobody and the tenant_id filter is
    the whole guard, so a known device_id from the wrong tenant must refuse
    rather than repatch.
    """
    squatter = TenantContext(
        tenant_id=str(uuid.uuid4()),
        database_url=URL,
        schema=tenants[0].schema,  # same schema, different tenant
    )
    mine = OutputStore(pool, tenants[0])
    theirs = OutputStore(pool, squatter)
    saved = await mine.save_device(a_device())

    assert await theirs.get_device(saved.device_id) is None
    assert await theirs.list_devices() == []
    assert await theirs.delete_device(saved.device_id) is False

    with pytest.raises(LookupError):
        await theirs.save_device(a_device(device_id=saved.device_id, name="Hijack"))

    still = await mine.get_device(saved.device_id)
    assert still is not None
    assert still.name == "Main hall DVS"
    assert still.languages == ["en", "fr", "ar"]


# --- constraints the database enforces --------------------------------------


async def test_two_devices_cannot_share_a_name_for_one_tenant(pool, tenants):
    """Operators pick by name; an ambiguity there is one the audience pays for."""
    store = OutputStore(pool, tenants[0])
    await store.save_device(a_device(name="Main hall DVS"))

    with pytest.raises(DuplicateDeviceName, match="already called 'Main hall DVS'"):
        await store.save_device(a_device(name="Main hall DVS"))


async def test_two_tenants_may_each_have_a_main_hall(pool, tenants):
    """The name is unique per tenant, not globally."""
    await OutputStore(pool, tenants[0]).save_device(a_device(name="Main hall DVS"))
    other = await OutputStore(pool, tenants[1]).save_device(a_device(name="Main hall DVS"))
    assert other.device_id


async def test_migration_is_idempotent(pool, tenants):
    """It runs on every start-up, so a second pass must be a no-op."""
    assert await migrate.apply_tenant(pool, tenants[0].schema) == []
