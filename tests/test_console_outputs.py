"""
Hardware output in the operator console.

The five operations are tested in tests/test_admin_api.py against the portal
door. This file is about the console door: a Google cookie instead of a
bearer token, a tenant fixed by the box instead of named in a header, and the
three "nothing here" states the page has to tell apart.

Skipped when Postgres is unreachable, for the same reason as the admin tests:
the handlers resolve a real tenant and write a real map, and a mock would test
the mock. Run tools/pg.sh start.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from lad_translate.console import auth
from lad_translate.console.app import create_app
from lad_translate.db import migrate

URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
CONTROL = "lad_test_control"
PUBLIC = "https://join.example.test"
BASE = "/console/api/outputs"

pytestmark = pytest.mark.asyncio

AUTH = auth.Config(
    client_id="cid", client_secret="csecret",
    redirect_uri="https://host/console/auth/callback",
    session_secret="test-secret",
    allowed_emails=frozenset(), allowed_domains=frozenset({"techiemaya.com"}),
)

SESSION_ENV = """LAD_CONTROL_SCHEMA=lad_test_control
LAD_TRANSLATE_TENANT=placeholder
LAD_TRANSLATE_TARGETS=fr,ar,de
"""


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(URL, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


def a_device(**over) -> dict:
    body = {
        "name": "Main hall DVS",
        "device_name": "Dante Virtual Soundcard",
        "channel_count": 16,
        "channels": [
            {"language": "en", "channel": 1, "label": "Floor"},
            {"language": "fr", "channel": 2, "ir_channel": 1},
            {"language": "ar", "channel": 3, "ir_channel": 2},
            {"language": "fr", "channel": 9, "gain_db": -6.0, "label": "Recorder"},
        ],
    }
    body.update(over)
    return body


def _signed_in(client) -> None:
    client.cookies.set(auth.COOKIE, auth.issue_session("op@techiemaya.com", AUTH.session_secret))


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "session.env"
    path.write_text(SESSION_ENV)
    return path


@pytest.fixture
async def db():
    """A pool, the control schema, and two tenants: ours and a bystander."""
    if not await _reachable():
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")
    import asyncpg

    pool = await asyncpg.create_pool(URL, min_size=1, max_size=4)
    await migrate.apply_control(pool, CONTROL)

    made = []
    for label in ("ours", "other"):
        tenant_id = str(uuid.uuid4())
        slug = f"con-{label}-{tenant_id[:8]}"
        schema = f"lad_con_{label}_{tenant_id.replace('-', '')[:8]}"
        await pool.execute(
            f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid,$2,$3)",
            tenant_id, slug, schema,
        )
        made.append({"id": tenant_id, "slug": slug, "schema": schema})

    yield pool, made

    for t in made:
        await pool.execute(f"DROP SCHEMA IF EXISTS {t['schema']} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", t["id"])
    await pool.close()


async def _client(env_file: Path, pool=None, tenant: str = "", control: str = CONTROL):
    from httpx import ASGITransport, AsyncClient

    app = create_app(
        public_base=PUBLIC, env_path=env_file, auth_config=AUTH,
        pool=pool, control_schema=control if pool is not None else None, tenant=tenant,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


# --- the door ----------------------------------------------------------------


async def test_anyone_not_signed_in_is_refused(env_file: Path):
    """The channel map decides which language reaches which wire."""
    async with await _client(env_file) as c:
        assert (await c.get(BASE)).status_code == 401
        assert (await c.post(f"{BASE}/devices", json=a_device())).status_code == 401


async def test_the_tenant_is_the_box_s_and_a_header_cannot_change_it(db, env_file: Path):
    """
    No X-Tenant-Id here, on purpose.

    The console serves exactly one tenant: the one in session.env, which is
    the one the session process writes to. A header naming another tenant is
    not an instruction; it is ignored, and the device lands where the box
    lives. The portal API is where a caller gets to name a tenant, and it
    holds a platform credential for the privilege.
    """
    pool, (ours, other) = db
    for t in (ours, other):
        await migrate.apply_tenant(pool, t["schema"])

    async with await _client(env_file, pool, ours["slug"]) as c:
        _signed_in(c)
        r = await c.post(
            f"{BASE}/devices", json=a_device(), headers={"X-Tenant-Id": other["id"]}
        )
        assert r.status_code == 201

    ours_rows = await pool.fetchval(f"SELECT count(*) FROM {ours['schema']}.audio_output_devices")
    other_rows = await pool.fetchval(f"SELECT count(*) FROM {other['schema']}.audio_output_devices")
    assert (ours_rows, other_rows) == (1, 0)


# --- the three empty states --------------------------------------------------


async def test_no_database_says_so_rather_than_showing_no_devices(env_file: Path):
    async with await _client(env_file) as c:
        _signed_in(c)
        r = await c.get(BASE)
        assert r.status_code == 200
        body = r.json()
        assert body["configured"] is False
        assert body["devices"] == []
        assert "LAD_DATABASE_URL" in body["reason"]
        # And the page still gets the language list, so it can render the
        # dropdown for when the box IS configured.
        assert [lang["code"] for lang in body["languages"]] == ["en", "fr", "ar", "de"]

        r = await c.post(f"{BASE}/devices", json=a_device())
        assert r.status_code == 503
        assert "LAD_DATABASE_URL" in r.json()["detail"]


async def test_an_unseeded_tenant_names_the_command_to_run(db, env_file: Path):
    pool, _ = db
    async with await _client(env_file, pool, "never-seeded") as c:
        _signed_in(c)
        body = (await c.get(BASE)).json()
        assert body["configured"] is True
        assert body["tenant"] is None
        assert "seed_tenant.py --slug never-seeded" in body["reason"]

        r = await c.post(f"{BASE}/devices", json=a_device())
        assert r.status_code == 503
        assert "seed_tenant.py" in r.json()["detail"]


async def test_a_tenant_without_migration_002_is_reported_not_empty(db, env_file: Path):
    """
    The state this whole module exists to distinguish.

    A schema that has never had the outputs tables answers every query with
    "relation does not exist". Swallowed, that is an empty device list on a
    page that offers to add one - and the add then fails with a stack trace.
    Reported, it is one line telling the operator what to run.
    """
    pool, (ours, _) = db
    await migrate.apply(pool, ours["schema"], ("001_translation_sessions.sql",))

    async with await _client(env_file, pool, ours["slug"]) as c:
        _signed_in(c)
        body = (await c.get(BASE)).json()
        assert body["configured"] is True
        assert body["migrated"] is False
        assert body["tenant"]["schema"] == ours["schema"]
        assert "--migrate-only" in body["reason"]

        r = await c.post(f"{BASE}/devices", json=a_device())
        assert r.status_code == 503
        assert "--migrate-only" in r.json()["detail"]


# --- the map -----------------------------------------------------------------


@pytest.fixture
async def ready(db, env_file: Path):
    pool, (ours, _) = db
    await migrate.apply_tenant(pool, ours["schema"])
    async with await _client(env_file, pool, ours["slug"]) as c:
        _signed_in(c)
        yield c


async def test_create_list_get_replace_delete(ready, caplog):
    # INFO on, so the handlers' log lines are actually built. A save that
    # writes the row and then dies constructing its own log record answers
    # 500 to the operator, and with logging off the test cannot see it -
    # which is exactly how extra={"name": ...} (a reserved LogRecord field)
    # got past this suite and was found in a browser.
    import logging

    caplog.set_level(logging.INFO)
    c = ready
    r = await c.post(f"{BASE}/devices", json=a_device())
    assert r.status_code == 201, r.text
    device = r.json()
    device_id = device["device_id"]
    assert sorted(device["languages"]) == ["ar", "en", "fr"]
    assert [ch["channel"] for ch in device["channels"]] == [1, 2, 3, 9]

    body = (await c.get(BASE)).json()
    assert body["migrated"] is True
    assert [d["device_id"] for d in body["devices"]] == [device_id]

    r = await c.get(f"{BASE}/devices/{device_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "Main hall DVS"

    # PUT replaces the WHOLE map: three channels in, three channels out, and
    # the recorder channel that was not resent is gone.
    repatched = a_device(channels=[
        {"language": "fr", "channel": 2, "ir_channel": 1},
        {"language": "ar", "channel": 3, "ir_channel": 2},
        {"language": "de", "channel": 4, "ir_channel": 3},
    ])
    r = await c.put(f"{BASE}/devices/{device_id}", json=repatched)
    assert r.status_code == 200, r.text
    assert [ch["channel"] for ch in r.json()["channels"]] == [2, 3, 4]

    assert (await c.delete(f"{BASE}/devices/{device_id}")).status_code == 204
    assert (await c.get(f"{BASE}/devices/{device_id}")).status_code == 404
    assert (await c.get(BASE)).json()["devices"] == []


async def test_put_edits_and_does_not_create(ready):
    r = await ready.put(f"{BASE}/devices/{uuid.uuid4()}", json=a_device())
    assert r.status_code == 404


async def test_invariants_are_refused_in_the_operator_s_words(ready):
    """The messages come from config.OutputDevice, not a generic 400."""
    two_on_one_wire = a_device(channels=[
        {"language": "fr", "channel": 2},
        {"language": "ar", "channel": 2},
    ])
    r = await ready.post(f"{BASE}/devices", json=two_on_one_wire)
    assert r.status_code == 422
    assert "share a channel" in r.json()["detail"]

    two_on_one_handset = a_device(channels=[
        {"language": "fr", "channel": 2, "ir_channel": 1},
        {"language": "ar", "channel": 3, "ir_channel": 1},
    ])
    r = await ready.post(f"{BASE}/devices", json=two_on_one_handset)
    assert r.status_code == 422
    assert "IR channel" in r.json()["detail"]

    beyond_the_card = a_device(channel_count=4, channels=[{"language": "fr", "channel": 9}])
    r = await ready.post(f"{BASE}/devices", json=beyond_the_card)
    assert r.status_code == 422
    assert "beyond the 4 channels" in r.json()["detail"]


async def test_a_duplicate_name_is_a_conflict(ready):
    assert (await ready.post(f"{BASE}/devices", json=a_device())).status_code == 201
    r = await ready.post(f"{BASE}/devices", json=a_device())
    assert r.status_code == 409


# --- signage -----------------------------------------------------------------


async def test_signage_lists_ir_channels_in_order_and_nothing_else(ready):
    """
    What goes on the wall by the handset table.

    The recorder feed (no IR number) and a disabled channel must not appear:
    neither is something an audience member can select on a handset.
    """
    device = a_device(channels=[
        {"language": "ar", "channel": 3, "ir_channel": 2},
        {"language": "fr", "channel": 2, "ir_channel": 1},
        {"language": "fr", "channel": 9, "label": "Recorder"},
        {"language": "de", "channel": 4, "ir_channel": 3, "enabled": False},
    ])
    r = await ready.post(f"{BASE}/devices", json=device)
    device_id = r.json()["device_id"]

    r = await ready.get(f"{BASE}/devices/{device_id}/signage")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    lines = [line for line in r.text.splitlines() if line.strip()]
    assert lines[0].startswith("Main hall DVS")
    assert lines[1:] == ["   1   Français  (French)", "   2   العربية  (Arabic)"]


async def test_signage_for_a_device_with_no_ir_channels_says_so(ready):
    r = await ready.post(
        f"{BASE}/devices", json=a_device(channels=[{"language": "en", "channel": 1}])
    )
    r = await ready.get(f"{BASE}/devices/{r.json()['device_id']}/signage")
    assert "No IR channels" in r.text
