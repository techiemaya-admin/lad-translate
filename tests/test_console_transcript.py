"""
The console's live transcript.

The grouping is where the bugs would be - one row per chunk per language,
collapsed to one line per chunk - so that is tested directly. The endpoint
is tested for the states an operator meets: no database, no session yet, a
session with lines, and the high-water mark that keeps a long talk from
being re-sent every two seconds.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lad_translate.console import auth
from lad_translate.console.app import create_app
from lad_translate.console.transcript import group_by_chunk

PUBLIC = "https://join.example.test"
AUTH = auth.Config(
    client_id="cid", client_secret="csecret",
    redirect_uri="https://host/console/auth/callback",
    session_secret="test-secret",
    allowed_emails=frozenset(), allowed_domains=frozenset({"techiemaya.com"}),
)
URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
CONTROL = "lad_test_control"


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "session.env"
    path.write_text("LAD_TRANSLATE_TARGETS=fr,ar\n")
    return path


def signed_in(client: TestClient) -> TestClient:
    client.cookies.set(auth.COOKIE, auth.issue_session("op@techiemaya.com", AUTH.session_secret))
    return client


# --- grouping ----------------------------------------------------------------


def row(chunk_id, language, source, translated, latency=1.0, start=0.0, end=1.0):
    return {
        "chunk_id": chunk_id, "language": language, "source_text": source,
        "translated_text": translated, "t_audio_start": start, "t_audio_end": end,
        "latency_s": latency,
    }


def test_one_line_per_chunk_with_every_language_beside_it():
    grouped = group_by_chunk([
        row(1, "ar", "Good morning", "صباح الخير", latency=1.36),
        row(1, "fr", "Good morning", "Bonjour", latency=1.21),
        row(2, "fr", "Welcome", "Bienvenue", latency=0.8),
    ])
    assert [c["chunk_id"] for c in grouped] == [1, 2]
    assert grouped[0]["source"] == "Good morning"
    assert set(grouped[0]["languages"]) == {"ar", "fr"}
    assert grouped[0]["languages"]["fr"]["text"] == "Bonjour"
    assert grouped[0]["languages"]["ar"]["latency_s"] == 1.36
    assert set(grouped[1]["languages"]) == {"fr"}


def test_a_language_that_produced_nothing_still_appears():
    """Three languages out of four is what an operator has to be able to see."""
    grouped = group_by_chunk([
        row(5, "fr", "Hello", "Bonjour"),
        row(5, "ar", "Hello", None, latency=None),
    ])
    assert grouped[0]["languages"]["ar"]["text"] is None
    assert grouped[0]["languages"]["ar"]["latency_s"] is None


def test_chunks_come_back_in_order_however_the_rows_arrived():
    grouped = group_by_chunk([row(9, "fr", "c", "c"), row(2, "fr", "a", "a"), row(5, "fr", "b", "b")])
    assert [c["chunk_id"] for c in grouped] == [2, 5, 9]


def test_no_rows_is_no_chunks():
    assert group_by_chunk([]) == []


# --- the endpoint -------------------------------------------------------------


def test_without_a_database_it_says_so_rather_than_looking_empty(env_file: Path):
    client = signed_in(TestClient(create_app(public_base=PUBLIC, env_path=env_file, auth_config=AUTH)))
    body = client.get("/console/api/transcript", params={"room": "hall-a"}).json()
    assert body["chunks"] == []
    assert "LAD_DATABASE_URL" in body["reason"]


def test_anonymous_callers_get_nothing(env_file: Path):
    anonymous = TestClient(create_app(public_base=PUBLIC, env_path=env_file, auth_config=AUTH))
    assert anonymous.get("/console/api/transcript").status_code == 401


def test_a_room_name_that_is_not_one_is_refused(env_file: Path):
    client = signed_in(TestClient(create_app(public_base=PUBLIC, env_path=env_file, auth_config=AUTH)))
    assert client.get("/console/api/transcript", params={"room": "a; rm -rf /"}).status_code == 400


@pytest.mark.asyncio
async def test_it_reads_a_real_session_and_honours_the_high_water_mark(env_file: Path):
    """
    The whole path: a session and its transcript rows in a tenant schema,
    read back through the console with `after` doing its job.
    """
    try:
        import asyncpg

        pool = await asyncpg.create_pool(URL, min_size=1, max_size=3, timeout=3)
    except Exception:
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")

    from lad_translate.config import (
        BackendSelection,
        LanguageTarget,
        SessionConfig,
        TenantContext,
    )
    from lad_translate.db import migrate
    from lad_translate.db.sessions import SessionStore, TranscriptRow

    tenant_id = str(uuid.uuid4())
    schema = f"lad_tr_{tenant_id.replace('-', '')[:8]}"
    slug = f"tr-{tenant_id[:8]}"
    await migrate.apply_control(pool, CONTROL)
    await pool.execute(
        f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid,$2,$3)",
        tenant_id, slug, schema,
    )
    await migrate.apply_tenant(pool, schema)
    try:
        tenant = TenantContext(tenant_id=tenant_id, database_url=URL, schema=schema)
        store = SessionStore(pool, tenant)
        config = SessionConfig(
            session_id=str(uuid.uuid4()), tenant=tenant, room_name="hall-a",
            event_name="Keynote", source_language="en",
            targets=[LanguageTarget("fr", "fr_FR-siwis-medium"), LanguageTarget("ar", "ar_JO-kareem-medium")],
            backends=BackendSelection(),
        )
        await store.create_session(config, latency_credible=False)
        await store.mark_live(config.session_id)
        for chunk_id, (src, fr, ar) in enumerate([
            ("Good morning everyone", "Bonjour à tous", "صباح الخير للجميع"),
            ("Welcome to Dubai", "Bienvenue à Dubaï", "مرحبا بكم في دبي"),
        ]):
            for lang, text in (("fr", fr), ("ar", ar)):
                await store.record_transcript(config.session_id, TranscriptRow(
                    chunk_id=chunk_id, language=lang, source_text=src, translated_text=text,
                    t_audio_start=chunk_id * 3.0, t_audio_end=chunk_id * 3.0 + 2.5,
                    latency_s=1.2, commit_reason="silence", revised=False,
                ))

        # httpx over ASGITransport, not TestClient: the pool belongs to this
        # test's event loop and TestClient drives the app on its own.
        from httpx import ASGITransport, AsyncClient

        app = create_app(
            public_base=PUBLIC, env_path=env_file, auth_config=AUTH,
            pool=pool, control_schema=CONTROL, tenant=slug,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            client.cookies.set(auth.COOKIE, auth.issue_session("op@techiemaya.com", AUTH.session_secret))

            body = (await client.get("/console/api/transcript", params={"room": "hall-a"})).json()
            assert body["session_id"] == config.session_id
            assert body["status"] == "live" and body["event_name"] == "Keynote"
            assert body["reason"] == ""
            assert [c["chunk_id"] for c in body["chunks"]] == [0, 1]
            assert body["chunks"][0]["source"] == "Good morning everyone"
            assert body["chunks"][0]["languages"]["ar"]["text"] == "صباح الخير للجميع"
            assert body["chunks"][1]["languages"]["fr"]["text"] == "Bienvenue à Dubaï"

            # The high-water mark: nothing newer than chunk 1.
            r = await client.get("/console/api/transcript", params={"room": "hall-a", "after": 1})
            assert r.json()["chunks"] == []
            # And only the newer half when asked from the middle.
            r = await client.get("/console/api/transcript", params={"room": "hall-a", "after": 0})
            assert [c["chunk_id"] for c in r.json()["chunks"]] == [1]

            # A room nobody has used says so.
            empty = (await client.get("/console/api/transcript", params={"room": "hall-b"})).json()
            assert empty["chunks"] == [] and "no session has run" in empty["reason"]
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", tenant_id)
        await pool.close()
