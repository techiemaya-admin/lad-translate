"""
Operator API for hardware output.

Skipped when Postgres is unreachable: the handlers resolve a real tenant and
write a real patch, and mocking that out would test the mock.

The authentication tests are the point of this file. This endpoint decides
which language reaches which wire, and it is reachable by anything that can
route to the port.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lad_translate.api.admin import create_admin_app
from lad_translate.db import migrate

URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
CONTROL = "lad_test_control"
TOKEN = "test-operator-token-not-a-real-secret"

pytestmark = pytest.mark.asyncio


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(URL, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


def a_device_body(**over) -> dict:
    body = {
        "name": "Main hall DVS",
        "device_name": "Dante Virtual Soundcard",
        "channel_count": 16,
        "channels": [
            {"language": "en", "channel": 1, "label": "Floor"},
            {"language": "fr", "channel": 2, "ir_channel": 1, "label": "Français"},
            {"language": "ar", "channel": 3, "ir_channel": 2, "label": "العربية"},
        ],
    }
    body.update(over)
    return body


@pytest.fixture
async def env():
    """A client, an authenticated header set, and two tenants to keep apart."""
    if not await _reachable():
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")
    import asyncpg
    from httpx import ASGITransport, AsyncClient

    pool = await asyncpg.create_pool(URL, min_size=1, max_size=4)
    await migrate.apply_control(pool, CONTROL)

    made = []
    for label in ("a", "b"):
        tenant_id = str(uuid.uuid4())
        schema = f"lad_adm_{label}_{tenant_id.replace('-', '')[:8]}"
        await pool.execute(
            f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid,$2,$3)",
            tenant_id, f"adm-{label}-{tenant_id[:8]}", schema,
        )
        await migrate.apply_tenant(pool, schema)
        made.append((tenant_id, schema))

    app = create_admin_app(pool=pool, control_schema=CONTROL, admin_token=TOKEN)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        yield client, made

    for tenant_id, schema in made:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", tenant_id)
    await pool.close()


def auth(tenant_id: str) -> dict:
    return {"Authorization": f"Bearer {TOKEN}", "X-Tenant-Id": tenant_id}


# --- authentication ---------------------------------------------------------


async def test_no_token_is_refused(env):
    client, made = env
    r = await client.get(
        "/api/admin/outputs/devices", headers={"X-Tenant-Id": made[0][0]}
    )
    assert r.status_code == 401


async def test_a_wrong_token_is_refused(env):
    client, made = env
    r = await client.get(
        "/api/admin/outputs/devices",
        headers={"Authorization": "Bearer nope", "X-Tenant-Id": made[0][0]},
    )
    assert r.status_code == 401


async def test_a_non_bearer_scheme_is_refused(env):
    client, made = env
    r = await client.get(
        "/api/admin/outputs/devices",
        headers={"Authorization": f"Basic {TOKEN}", "X-Tenant-Id": made[0][0]},
    )
    assert r.status_code == 401


async def test_an_unconfigured_service_refuses_everything():
    """
    No LAD_ADMIN_TOKEN must disable the API, not open it. A default here would
    be a working password for anyone who read the source.
    """
    from httpx import ASGITransport, AsyncClient

    app = create_admin_app(pool=None, control_schema=CONTROL, admin_token="")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get(
            "/api/admin/outputs/devices",
            headers={"Authorization": "Bearer anything", "X-Tenant-Id": str(uuid.uuid4())},
        )
    assert r.status_code == 503


async def test_a_valid_token_without_a_tenant_is_refused(env):
    client, _ = env
    r = await client.get(
        "/api/admin/outputs/devices", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert r.status_code == 400


async def test_an_unknown_tenant_is_not_found(env):
    client, _ = env
    r = await client.get("/api/admin/outputs/devices", headers=auth(str(uuid.uuid4())))
    assert r.status_code == 404


# --- CRUD -------------------------------------------------------------------


async def test_create_then_read_back(env):
    client, made = env
    tenant_id = made[0][0]

    created = await client.post(
        "/api/admin/outputs/devices", json=a_device_body(), headers=auth(tenant_id)
    )
    assert created.status_code == 201
    device_id = created.json()["device_id"]
    assert created.json()["languages"] == ["en", "fr", "ar"]

    got = await client.get(f"/api/admin/outputs/devices/{device_id}", headers=auth(tenant_id))
    assert got.status_code == 200
    assert got.json()["channels"][1] == {
        "language": "fr", "channel": 2, "ir_channel": 1,
        "label": "Français", "gain_db": 0.0, "enabled": True,
    }


async def test_put_replaces_the_whole_map(env):
    client, made = env
    tenant_id = made[0][0]
    device_id = (
        await client.post(
            "/api/admin/outputs/devices", json=a_device_body(), headers=auth(tenant_id)
        )
    ).json()["device_id"]

    replaced = await client.put(
        f"/api/admin/outputs/devices/{device_id}",
        json=a_device_body(channels=[{"language": "hi", "channel": 1, "ir_channel": 4}]),
        headers=auth(tenant_id),
    )

    assert replaced.status_code == 200
    assert replaced.json()["languages"] == ["hi"]
    assert len(replaced.json()["channels"]) == 1


async def test_delete_then_gone(env):
    client, made = env
    tenant_id = made[0][0]
    device_id = (
        await client.post(
            "/api/admin/outputs/devices", json=a_device_body(), headers=auth(tenant_id)
        )
    ).json()["device_id"]

    assert (
        await client.delete(f"/api/admin/outputs/devices/{device_id}", headers=auth(tenant_id))
    ).status_code == 204
    assert (
        await client.get(f"/api/admin/outputs/devices/{device_id}", headers=auth(tenant_id))
    ).status_code == 404


# --- validation reaches the operator ----------------------------------------


async def test_a_channel_beyond_the_device_is_rejected_with_a_usable_message(env):
    client, made = env
    r = await client.post(
        "/api/admin/outputs/devices",
        json=a_device_body(
            channel_count=8, channels=[{"language": "fr", "channel": 12}]
        ),
        headers=auth(made[0][0]),
    )
    assert r.status_code == 422
    assert "beyond the 8 channels" in r.json()["detail"]


async def test_two_languages_on_one_wire_is_rejected(env):
    client, made = env
    r = await client.post(
        "/api/admin/outputs/devices",
        json=a_device_body(
            channels=[
                {"language": "fr", "channel": 3},
                {"language": "ar", "channel": 3},
            ]
        ),
        headers=auth(made[0][0]),
    )
    assert r.status_code == 422
    assert "share a channel" in r.json()["detail"]


async def test_two_languages_on_one_ir_channel_is_rejected(env):
    client, made = env
    r = await client.post(
        "/api/admin/outputs/devices",
        json=a_device_body(
            channels=[
                {"language": "fr", "channel": 1, "ir_channel": 5},
                {"language": "ar", "channel": 2, "ir_channel": 5},
            ]
        ),
        headers=auth(made[0][0]),
    )
    assert r.status_code == 422
    assert "share an IR channel" in r.json()["detail"]


async def test_a_rate_dante_does_not_run_is_rejected(env):
    client, made = env
    r = await client.post(
        "/api/admin/outputs/devices",
        json=a_device_body(sample_rate=22050),
        headers=auth(made[0][0]),
    )
    assert r.status_code == 422


# --- tenant isolation -------------------------------------------------------


async def test_a_device_is_invisible_to_another_tenant(env):
    client, made = env
    device_id = (
        await client.post(
            "/api/admin/outputs/devices", json=a_device_body(), headers=auth(made[0][0])
        )
    ).json()["device_id"]

    assert (
        await client.get(f"/api/admin/outputs/devices/{device_id}", headers=auth(made[1][0]))
    ).status_code == 404
    assert (
        await client.get("/api/admin/outputs/devices", headers=auth(made[1][0]))
    ).json()["devices"] == []


async def test_another_tenant_cannot_repatch_a_known_device_id(env):
    """A valid token plus a known id must still not cross the tenant boundary."""
    client, made = env
    device_id = (
        await client.post(
            "/api/admin/outputs/devices", json=a_device_body(), headers=auth(made[0][0])
        )
    ).json()["device_id"]

    hijack = await client.put(
        f"/api/admin/outputs/devices/{device_id}",
        json=a_device_body(name="Hijacked"),
        headers=auth(made[1][0]),
    )
    assert hijack.status_code == 404

    still = await client.get(
        f"/api/admin/outputs/devices/{device_id}", headers=auth(made[0][0])
    )
    assert still.json()["name"] == "Main hall DVS"


async def test_another_tenant_cannot_delete_a_known_device_id(env):
    client, made = env
    device_id = (
        await client.post(
            "/api/admin/outputs/devices", json=a_device_body(), headers=auth(made[0][0])
        )
    ).json()["device_id"]

    assert (
        await client.delete(f"/api/admin/outputs/devices/{device_id}", headers=auth(made[1][0]))
    ).status_code == 404
    assert (
        await client.get(f"/api/admin/outputs/devices/{device_id}", headers=auth(made[0][0]))
    ).status_code == 200


async def test_a_duplicate_device_name_is_a_conflict_not_a_crash(env):
    """Reusing a name is an ordinary slip; it must not surface as a 500."""
    client, made = env
    tenant_id = made[0][0]
    await client.post(
        "/api/admin/outputs/devices", json=a_device_body(), headers=auth(tenant_id)
    )

    again = await client.post(
        "/api/admin/outputs/devices", json=a_device_body(), headers=auth(tenant_id)
    )

    assert again.status_code == 409
    assert "already called" in again.json()["detail"]


async def test_renaming_onto_another_device_is_a_conflict(env):
    client, made = env
    tenant_id = made[0][0]
    await client.post(
        "/api/admin/outputs/devices", json=a_device_body(name="Main hall DVS"), headers=auth(tenant_id)
    )
    second = await client.post(
        "/api/admin/outputs/devices", json=a_device_body(name="Breakout DVS"), headers=auth(tenant_id)
    )

    clash = await client.put(
        f"/api/admin/outputs/devices/{second.json()['device_id']}",
        json=a_device_body(name="Main hall DVS"),
        headers=auth(tenant_id),
    )
    assert clash.status_code == 409
