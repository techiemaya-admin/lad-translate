"""
Listener join API.

Runs against a real Postgres, like test_db_sessions. Skipped when there is none.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lad_translate.api.join import create_app
from lad_translate.api.tokens import TokenIssuer
from lad_translate.config import LanguageTarget, SessionConfig, TenantContext
from lad_translate.db import migrate
from lad_translate.db.sessions import SessionStore

URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
CONTROL = "lad_join_control"

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
async def env():
    if not await _reachable():
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")
    import asyncpg
    from httpx import ASGITransport, AsyncClient

    pool = await asyncpg.create_pool(URL, min_size=1, max_size=4)
    await migrate.apply_control(pool, CONTROL)

    tenant_id = str(uuid.uuid4())
    schema = f"lad_join_{tenant_id.replace('-', '')[:8]}"
    await pool.execute(
        f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid,$2,$3)",
        tenant_id, f"join-{tenant_id[:8]}", schema,
    )
    await migrate.apply_tenant(pool, schema)

    tenant = TenantContext(tenant_id=tenant_id, database_url=URL, schema=schema)
    config = SessionConfig(
        session_id=str(uuid.uuid4()), tenant=tenant, room_name="room-x",
        event_name="Sharjah Summit", source_language="en",
        targets=[LanguageTarget("fr", "v-fr"), LanguageTarget("ar", "v-ar")],
    )
    store = SessionStore(pool, tenant)
    await store.create_session(config, latency_credible=False)
    await store.mark_live(config.session_id)

    issuer = TokenIssuer(
        livekit_url="ws://127.0.0.1:7880", api_key="devkey",
        api_secret="lad-test-signing-key-not-a-real-secret",
    )
    app = create_app(pool=pool, issuer=issuer, control_schema=CONTROL)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        yield client, store, config

    await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", tenant_id)
    await pool.close()


# --- session info -----------------------------------------------------------


async def test_session_info_lists_languages_in_their_own_script(env):
    """The person choosing reads that language, not necessarily English."""
    client, _store, config = env
    r = await client.get(f"/api/sessions/{config.session_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["event_name"] == "Sharjah Summit"
    by_code = {lang["code"]: lang for lang in body["languages"]}
    assert by_code["fr"]["native"] == "Français"
    assert by_code["ar"]["native"] == "العربية"
    assert by_code["ar"]["rtl"] is True
    assert by_code["fr"]["rtl"] is False


async def test_unknown_session_is_a_404(env):
    client, *_ = env
    assert (await client.get(f"/api/sessions/{uuid.uuid4()}")).status_code == 404


async def test_ended_session_is_gone_not_missing(env):
    """410 rather than 404, so the page can say 'ended' not 'check the QR code'."""
    client, store, config = env
    await store.end_session(config.session_id)
    assert (await client.get(f"/api/sessions/{config.session_id}")).status_code == 410


# --- joining ----------------------------------------------------------------


async def test_join_issues_a_token_and_names_one_track(env):
    client, _store, config = env
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "fr"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "fr"
    assert body["track_name"] == "lang-fr"
    assert body["token"]
    assert body["url"].startswith("ws")


async def test_join_records_the_listener(env):
    client, store, config = env
    await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "fr"})
    await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "ar"})
    assert await store.active_listeners(config.session_id) == {"fr": 1, "ar": 1}


async def test_a_language_not_on_offer_is_refused(env):
    """Otherwise a listener gets a valid token for a track that does not exist."""
    client, _store, config = env
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "zh"})
    assert r.status_code == 400
    assert "not offered" in r.json()["detail"]


async def test_joining_an_ended_session_is_refused(env):
    client, store, config = env
    await store.end_session(config.session_id)
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "fr"})
    assert r.status_code == 410


async def test_join_body_is_read_as_json_not_a_query_parameter(env):
    """
    Regression guard.

    This module uses `from __future__ import annotations`, so FastAPI resolves
    handler annotations against module globals. Defining the request model
    inside create_app made it invisible there, and the body silently degraded
    to a required query parameter: every join returned 422.
    """
    client, _store, config = env
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "fr"})
    assert r.status_code != 422, "request body is not being parsed as JSON"


# --- leaving ----------------------------------------------------------------


async def test_leave_clears_the_listener(env):
    client, store, config = env
    joined = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "fr"})
    listener_id = joined.json()["listener_id"]
    r = await client.post(
        f"/api/listeners/{listener_id}/leave", params={"session_id": config.session_id}
    )
    assert r.status_code == 200
    assert await store.active_listeners(config.session_id) == {}


# --- page -------------------------------------------------------------------


async def test_join_page_is_served(env):
    client, _store, config = env
    r = await client.get(f"/s/{config.session_id}")
    assert r.status_code == 200
    assert "Live Translation" in r.text


async def test_livekit_client_is_served_locally_not_from_a_cdn(env):
    """
    Venue wifi routinely blocks third party origins. A join page that cannot
    fetch its own client library is a room full of people hearing nothing.
    """
    client, *_ = env
    page = (await client.get(f"/s/{uuid.uuid4()}")).text
    assert "/static/vendor/livekit-client.umd.js" in page
    assert "cdn." not in page and "unpkg" not in page
    assert (await client.get("/static/vendor/livekit-client.umd.js")).status_code == 200


async def test_healthz(env):
    client, *_ = env
    assert (await client.get("/healthz")).json() == {"ok": True}


# --- original audio ---------------------------------------------------------


async def test_the_original_language_is_offered_first(env):
    """
    Someone in a hall with poor acoustics, or hard of hearing, wants the floor
    audio in their earbuds more than anything downstream of it.
    """
    client, _store, config = env
    body = (await client.get(f"/api/sessions/{config.session_id}")).json()
    first = body["languages"][0]
    assert first["code"] == "en"
    assert first["is_source"] is True
    assert first["english"] == "original audio"


async def test_the_original_maps_to_the_source_track_not_a_translation(env):
    """
    A relay, not a translation. Sending English through STT, translation and
    synthesis would add transcription errors, a synthetic voice and seconds of
    latency to audio that is already perfect.
    """
    client, _store, config = env
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["track_name"] == "source-audio"
    assert body["is_source"] is True


async def test_translations_still_map_to_their_own_tracks(env):
    client, _store, config = env
    body = (await client.post(
        f"/api/sessions/{config.session_id}/join", json={"language": "fr"}
    )).json()
    assert body["track_name"] == "lang-fr"
    assert body["is_source"] is False


async def test_a_language_that_is_neither_source_nor_target_is_still_refused(env):
    client, _store, config = env
    r = await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "zh"})
    assert r.status_code == 400
    assert "en" in r.json()["detail"], "the error should list the original as available"


async def test_listening_to_the_original_is_counted(env):
    """Listener counts drive the room load picture, and a relay listener is
    still a subscriber on the SFU."""
    client, store, config = env
    await client.post(f"/api/sessions/{config.session_id}/join", json={"language": "en"})
    assert (await store.active_listeners(config.session_id)).get("en") == 1
