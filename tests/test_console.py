"""
Operator console.

The console can restart sessions and rewrite settings, so the tests that matter
are the ones about what it REFUSES: keys outside the allowlist, room names that
would reach systemd as arguments, and QR codes built on a session id.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lad_translate.console import env, sessions
from lad_translate.console.app import create_app

PUBLIC = "https://join.example.test"

SAMPLE = """# a comment that must survive a write
LAD_CONTROL_SCHEMA=lad_translate_dev
LIVEKIT_URL=wss://sfu.example.test

# the measured pair - moving one alone is how audio gets shed
LAD_TRANSLATE_EMIT_INTERVAL=3.0
LAD_TRANSLATE_WINDOW=6.0
LAD_TRANSLATE_STT_MODEL=tiny
"""


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "session.env"
    path.write_text(SAMPLE)
    return path


@pytest.fixture
def client(env_file: Path) -> TestClient:
    return TestClient(create_app(public_base=PUBLIC, env_path=env_file))


# --- writing settings -------------------------------------------------------

def test_a_write_preserves_comments_and_order(env_file: Path):
    """
    The comments carry why each number is what it is - what was measured and
    what it cost. A console that regenerated the file from a template would
    throw that away on the first save, and those comments are most of what
    stops someone setting emit 1.0 at a venue.
    """
    env.write({"LAD_TRANSLATE_STT_MODEL": "small"}, env_file)
    text = env_file.read_text()

    assert "# a comment that must survive a write" in text
    assert "# the measured pair" in text
    assert "LAD_TRANSLATE_STT_MODEL=small" in text
    assert text.index("LAD_CONTROL_SCHEMA") < text.index("LAD_TRANSLATE_EMIT_INTERVAL")


def test_only_the_allowlist_is_writable(env_file: Path):
    """
    This file also holds the control schema and the LiveKit addresses. A console
    that can rewrite those can point a venue at the wrong SFU, so the guard is
    an allowlist and the refusal is loud rather than a silent skip.
    """
    with pytest.raises(ValueError, match="not editable"):
        env.write({"LIVEKIT_URL": "wss://attacker.example"}, env_file)
    assert "wss://sfu.example.test" in env_file.read_text()


def test_a_new_key_is_appended_not_dropped(env_file: Path):
    """Same reasoning as bootstrap: a release that adds a setting must reach a
    box that predates it, or the unit expands a variable to nothing."""
    changed = env.write({"LAD_TRANSLATE_LOOKAHEAD": "80ms"}, env_file)
    assert changed == ["LAD_TRANSLATE_LOOKAHEAD"]
    assert "LAD_TRANSLATE_LOOKAHEAD=80ms" in env_file.read_text()


def test_an_unchanged_value_reports_no_change(env_file: Path):
    assert env.write({"LAD_TRANSLATE_STT_MODEL": "tiny"}, env_file) == []


# --- room names reach systemd ----------------------------------------------

@pytest.mark.parametrize(
    "bad",
    ["../etc", "room name", "Room", "a;systemctl", "", "-leading", "x" * 64],
)
def test_a_room_name_that_could_reach_systemd_is_refused(bad: str):
    with pytest.raises(sessions.BadRoom):
        sessions.validate_room(bad)


@pytest.mark.parametrize("ok", ["dubai-demo", "hall-a", "keynote2", "a"])
def test_ordinary_room_names_are_accepted(ok: str):
    assert sessions.validate_room(ok) == ok


def test_the_api_refuses_a_bad_room_before_touching_systemd(client: TestClient):
    r = client.post("/api/apply", json={"room": "a;rm -rf /", "preset": "live-safe"})
    assert r.status_code == 400


# --- presets ----------------------------------------------------------------

def test_every_preset_carries_its_measurement(client: TestClient):
    """A preset without evidence is a slider with a nicer name."""
    presets = client.get("/api/presets").json()["presets"]
    assert presets
    for p in presets:
        assert p["measured"].strip(), f"{p['key']} has no measurement"


def test_the_dangerous_presets_carry_warnings(client: TestClient):
    """
    low-latency measured better on the fixture and dropped 225s of live speech;
    streaming has no VAD. If either ever loses its warning, the console is
    actively misleading rather than merely incomplete.
    """
    by_key = {p["key"]: p for p in client.get("/api/presets").json()["presets"]}
    for key in ("low-latency", "streaming"):
        assert by_key[key]["warning"], f"{key} must keep its warning"


def test_applying_a_preset_writes_its_values(client: TestClient, env_file: Path):
    r = client.post("/api/apply", json={"room": "hall-a", "preset": "accurate",
                                        "restart": False})
    assert r.status_code == 200
    settings = env.read(env_file)
    assert settings["LAD_TRANSLATE_STT_MODEL"] == "small"
    assert settings["LAD_TRANSLATE_EMIT_INTERVAL"] == "3.0"
    assert settings["LAD_TRANSLATE_WINDOW"] == "6.0"


def test_raw_settings_win_over_the_preset(client: TestClient, env_file: Path):
    """"Pick a preset, then adjust one field" has to do what it looks like."""
    client.post("/api/apply", json={
        "room": "hall-a", "preset": "accurate", "restart": False,
        "settings": {"LAD_TRANSLATE_STT_MODEL": "tiny"},
    })
    assert env.read(env_file)["LAD_TRANSLATE_STT_MODEL"] == "tiny"


def test_an_unknown_preset_is_refused(client: TestClient):
    r = client.post("/api/apply", json={"room": "hall-a", "preset": "fastest",
                                        "restart": False})
    assert r.status_code == 400


# --- QR codes ---------------------------------------------------------------

def test_qr_encodes_a_room_url_not_a_session_id(client: TestClient):
    """
    A session id dies with the session. A printed code built on one is a wall
    of paper pointing at a 404 the first time a worker restarts, which is a
    failure that happens at a venue and cannot be fixed by reprinting.
    """
    r = client.get("/api/qr", params={"room": "dubai-demo", "kind": "listen"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["X-Encoded-Url"] == f"{PUBLIC}/room/dubai-demo"


def test_the_speaker_code_points_at_the_publish_page(client: TestClient):
    r = client.get("/api/qr", params={"room": "dubai-demo", "kind": "speak"})
    assert r.headers["X-Encoded-Url"] == f"{PUBLIC}/room/dubai-demo/speak"


def test_qr_points_at_the_join_service_not_this_box(client: TestClient):
    """
    The console runs on the SFU host. A phone cannot reach it, and it is behind
    basic auth in any case, so a code pointing here is a code nobody can use.
    """
    url = client.get("/api/qr", params={"room": "hall-a"}).headers["X-Encoded-Url"]
    assert url.startswith(PUBLIC)


def test_qr_refuses_an_unknown_kind(client: TestClient):
    assert client.get("/api/qr", params={"room": "hall-a", "kind": "print"}).status_code == 400
