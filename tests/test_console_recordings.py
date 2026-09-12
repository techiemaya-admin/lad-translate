"""
Recording, from the console's side.

The REC button has to do two things at once - signal the running session and
set the flag for the next one - and the recordings endpoints have to serve
files from exactly one directory and nothing else. Those are the tests here.
The recorder itself is tested in tests/test_recording.py.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lad_translate.console import auth, env, sessions
from lad_translate.console.app import create_app
from lad_translate.console.sessions import _summarise

PUBLIC = "https://join.example.test"
AUTH = auth.Config(
    client_id="cid", client_secret="csecret",
    redirect_uri="https://host/console/auth/callback",
    session_secret="test-secret",
    allowed_emails=frozenset(), allowed_domains=frozenset({"techiemaya.com"}),
)
TAKE = "2f1c6c2e-1111-4222-8333-444455556666-20260912T081244Z"


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "session.env"
    path.write_text("LAD_TRANSLATE_TARGETS=fr,ar\nLAD_TRANSLATE_RECORD_FLAG=\n")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "recordings" / "hall-a" / TAKE
    base.mkdir(parents=True)
    (base / "manifest.json").write_text(json.dumps({
        "session_id": TAKE[:36], "room": "hall-a", "event_name": "Keynote",
        "started_at": "2026-09-12T08:12:44+00:00", "ended_at": "2026-09-12T08:42:44+00:00",
        "seconds": 1800.0, "files": {},
    }))
    (base / "source.wav").write_bytes(b"RIFF" + b"\x00" * 40 + b"\x01\x02" * 100)
    (base / "fr.wav").write_bytes(b"RIFF" + b"\x00" * 40 + b"\x03\x04" * 50)
    (base / "notes.txt").write_text("not a recording file")
    return tmp_path / "recordings"


@pytest.fixture
def client(env_file: Path, root: Path) -> TestClient:
    c = TestClient(create_app(public_base=PUBLIC, env_path=env_file, auth_config=AUTH,
                              recordings_root=root))
    c.cookies.set(auth.COOKIE, auth.issue_session("op@techiemaya.com", AUTH.session_secret))
    return c


@pytest.fixture
def signals(monkeypatch):
    """Capture what the console would run instead of running systemctl."""
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(sessions, "_run", fake_run)
    return calls


# --- the button ----------------------------------------------------------------


def test_rec_on_signals_the_session_and_sets_the_flag(client, env_file, signals):
    r = client.post("/console/api/record", json={"room": "hall-a", "on": True})
    assert r.status_code == 200
    assert env.read(env_file)["LAD_TRANSLATE_RECORD_FLAG"] == "--record"
    assert signals == [("sudo", "-n", "systemctl", "kill", "--signal=SIGUSR1",
                        "lad-translate-session@hall-a.service")]


def test_rec_off_clears_the_flag_and_sends_the_other_signal(client, env_file, signals):
    client.post("/console/api/record", json={"room": "hall-a", "on": True})
    r = client.post("/console/api/record", json={"room": "hall-a", "on": False})
    assert r.status_code == 200
    assert env.read(env_file)["LAD_TRANSLATE_RECORD_FLAG"] == ""
    assert signals[-1][4] == "--signal=SIGUSR2"


def test_a_room_name_that_is_not_one_never_reaches_systemctl(client, signals):
    r = client.post("/console/api/record", json={"room": "hall a; rm -rf /", "on": True})
    assert r.status_code == 400
    assert signals == []


@pytest.mark.asyncio
async def test_a_stopped_session_is_not_an_error_for_rec(monkeypatch):
    """The flag is set; the next start honours it. That is the contract."""

    async def not_running(*args):
        return 1, "Failed to kill unit: Unit lad-translate-session@hall-a.service not loaded."

    monkeypatch.setattr(sessions, "_run", not_running)
    await sessions.record("hall-a", True)  # does not raise


# --- status ----------------------------------------------------------------------


def test_status_reads_the_recording_state_from_the_journal():
    lines = [
        json.dumps({"severity": "INFO", "message": "session created", "session_id": "abc"}),
        json.dumps({"severity": "INFO", "message": "recording started",
                    "directory": "/var/lib/ladtranslate/recordings/hall-a/x", "take": 1}),
    ]
    s = _summarise("hall-a", True, "now", lines)
    assert s.recording is True
    assert s.recording_dir == "/var/lib/ladtranslate/recordings/hall-a/x"
    assert s.recording_takes == 1

    lines.append(json.dumps({"severity": "INFO", "message": "recording stopped", "seconds": 12.0}))
    s = _summarise("hall-a", True, "now", lines)
    assert s.recording is False and s.recording_dir is None


def test_a_stopped_unit_is_never_recording_whatever_the_journal_says():
    lines = [json.dumps({"severity": "INFO", "message": "recording started", "directory": "/x", "take": 1})]
    s = _summarise("hall-a", False, None, lines)
    assert s.recording is False


# --- the files --------------------------------------------------------------------


def test_takes_are_listed_with_only_recording_files(client):
    r = client.get("/console/api/recordings", params={"room": "hall-a"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["takes"]) == 1
    take = body["takes"][0]
    assert take["name"] == TAKE and take["session_id"] == TAKE[:36]
    assert take["seconds"] == 1800.0 and take["in_progress"] is False
    assert [f["name"] for f in take["files"]] == ["fr.wav", "manifest.json", "source.wav"]
    assert "notes.txt" not in str(take["files"])
    assert body["disk"]["free_bytes"] is not None


def test_a_room_with_no_recordings_is_an_empty_list(client):
    assert client.get("/console/api/recordings", params={"room": "hall-b"}).json()["takes"] == []


def test_a_file_downloads_as_audio(client):
    r = client.get(f"/console/api/recordings/hall-a/{TAKE}/fr.wav")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert "fr.wav" in r.headers["content-disposition"]
    assert r.content.startswith(b"RIFF")


def test_only_recording_shaped_names_are_served(client):
    """Path traversal is a shape mismatch. Refused before the filesystem."""
    assert client.get(f"/console/api/recordings/hall-a/{TAKE}/notes.txt").status_code == 400
    assert client.get(f"/console/api/recordings/hall-a/{TAKE}/..%2Fmanifest.json").status_code in (400, 404)
    assert client.get("/console/api/recordings/hall-a/..%2F..%2Fetc/passwd").status_code in (400, 404)
    assert client.get(f"/console/api/recordings/hall-a/{TAKE}/zz.wav").status_code == 404
    assert client.get(f"/console/api/recordings/..%2Fhall-a/{TAKE}/fr.wav").status_code in (400, 404)


def test_a_take_can_be_deleted_and_nothing_else_can(client, root):
    assert client.delete(f"/console/api/recordings/hall-a/{TAKE}").status_code == 204
    assert not (root / "hall-a" / TAKE).exists()
    assert (root / "hall-a").exists(), "the room directory stays"
    assert client.delete(f"/console/api/recordings/hall-a/{TAKE}").status_code == 404
    assert client.delete("/console/api/recordings/hall-a/not-a-take").status_code == 400


def test_anonymous_callers_get_nothing(env_file, root):
    anonymous = TestClient(create_app(public_base=PUBLIC, env_path=env_file, auth_config=AUTH,
                                      recordings_root=root))
    assert anonymous.get("/console/api/recordings").status_code == 401
    assert anonymous.get(f"/console/api/recordings/hall-a/{TAKE}/fr.wav").status_code == 401
    assert anonymous.post("/console/api/record", json={"room": "hall-a", "on": True}).status_code == 401


# --- the disclosure flag ---------------------------------------------------------------

URL = os.getenv("LAD_TEST_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent")
CONTROL = "lad_test_control"


@pytest.mark.asyncio
async def test_the_flag_round_trips_through_the_store():
    """The speaker page reads this row. It has to be there and tenant-scoped."""
    try:
        import asyncpg

        pool = await asyncpg.create_pool(URL, min_size=1, max_size=2, timeout=3)
    except Exception:
        pytest.skip(f"no Postgres at {URL}; run tools/pg.sh start")

    from lad_translate.config import (
        BackendSelection,
        LanguageTarget,
        SessionConfig,
        TenantContext,
    )
    from lad_translate.db import migrate
    from lad_translate.db.sessions import SessionStore

    tenant_id = str(uuid.uuid4())
    schema = f"lad_rec_{tenant_id.replace('-', '')[:8]}"
    await migrate.apply_control(pool, CONTROL)
    await pool.execute(
        f"INSERT INTO {CONTROL}.tenants (id, slug, schema_name) VALUES ($1::uuid,$2,$3)",
        tenant_id, f"rec-{tenant_id[:8]}", schema,
    )
    await migrate.apply_tenant(pool, schema)
    try:
        tenant = TenantContext(tenant_id=tenant_id, database_url=URL, schema=schema)
        store = SessionStore(pool, tenant)
        config = SessionConfig(
            session_id=str(uuid.uuid4()), tenant=tenant, room_name="hall-a",
            event_name="Keynote", source_language="en",
            targets=[LanguageTarget("fr", "fr_FR-siwis-medium")], backends=BackendSelection(),
        )
        await store.create_session(config, latency_credible=False)
        assert (await store.get_session(config.session_id))["recording"] is False
        await store.set_recording(config.session_id, True)
        assert (await store.get_session(config.session_id))["recording"] is True
        await store.set_recording(config.session_id, False)
        assert (await store.get_session(config.session_id))["recording"] is False
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.execute(f"DELETE FROM {CONTROL}.tenants WHERE id = $1::uuid", tenant_id)
        await pool.close()
