"""
Start, stop and inspect the session units.

The console runs as ladtranslate and shells out to systemctl through a narrow
sudoers rule. It is a web service: running it as root so it can manage units
would put a browser one bug away from the box.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from ..obs.log import get_logger

log = get_logger(__name__)

UNIT = "lad-translate-session@{room}.service"

# A room name reaches systemd and journalctl as part of a unit name, so it is
# validated rather than escaped. Anything outside this set is rejected before
# it becomes an argument.
ROOM = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class BadRoom(ValueError):
    pass


def validate_room(room: str) -> str:
    if not ROOM.match(room):
        raise BadRoom(
            "room must be lowercase letters, digits and hyphens, "
            "starting with a letter or digit, up to 63 characters"
        )
    return room


@dataclass(frozen=True)
class SessionStatus:
    room: str
    active: bool
    since: str | None
    session_id: str | None
    backend: str | None
    model: str | None
    chunks: int
    drops: int
    dropped_s: float
    skips: int
    errors: int
    latency_p50: float | None
    latency_max: float | None
    waiting_for_speaker: bool


async def _run(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


async def restart(room: str) -> None:
    validate_room(room)
    code, out = await _run("sudo", "-n", "systemctl", "restart", UNIT.format(room=room))
    if code != 0:
        raise RuntimeError(f"systemctl restart failed: {out.strip()[:300]}")
    log.info("session restarted", extra={"room": room})


async def stop(room: str) -> None:
    validate_room(room)
    code, out = await _run("sudo", "-n", "systemctl", "stop", UNIT.format(room=room))
    if code != 0:
        raise RuntimeError(f"systemctl stop failed: {out.strip()[:300]}")
    log.info("session stopped", extra={"room": room})


async def status(room: str) -> SessionStatus:
    validate_room(room)
    unit = UNIT.format(room=room)

    _, active_out = await _run("systemctl", "is-active", unit)
    active = active_out.strip() == "active"

    _, since_out = await _run("systemctl", "show", unit, "-p", "ActiveEnterTimestamp", "--value")
    since = since_out.strip() or None

    # Only this activation's journal. Without --since the counters accumulate
    # across restarts, which makes a fresh session look like it inherited the
    # last one's failures.
    lines: list[str] = []
    if since:
        _, journal = await _run(
            "journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "cat"
        )
        lines = journal.splitlines()

    return _summarise(room, active, since, lines)


def _summarise(room: str, active: bool, since: str | None, lines: list[str]) -> SessionStatus:
    session_id = backend = model = None
    chunks = drops = skips = errors = 0
    dropped_s = 0.0
    latencies: list[float] = []
    waiting = False

    for raw in lines:
        if "waiting for a speaker" in raw:
            waiting = True
        if not raw.startswith("{"):
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue

        message = entry.get("message", "")
        if entry.get("severity") == "ERROR":
            errors += 1
        if message == "chunk published":
            chunks += 1
            value = entry.get("glass_to_glass_s")
            if isinstance(value, (int, float)):
                latencies.append(float(value))
        elif message == "audio dropped to recover from backlog":
            drops += 1
            dropped_s = float(entry.get("total_seconds_dropped", dropped_s))
        elif message == "phrase skipped to recover playout drift":
            skips += 1
        elif message == "session created":
            session_id = entry.get("session_id") or session_id
        elif message.startswith("Whisper STT loaded"):
            backend, model = "faster-whisper", entry.get("model")
        elif message == "FastConformer loaded":
            backend, model = "fastconformer", entry.get("lookahead")

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    worst = latencies[-1] if latencies else None

    return SessionStatus(
        room=room,
        active=active,
        since=since,
        session_id=session_id,
        backend=backend,
        model=model,
        chunks=chunks,
        drops=drops,
        dropped_s=round(dropped_s, 1),
        skips=skips,
        errors=errors,
        latency_p50=round(p50, 2) if p50 is not None else None,
        latency_max=round(worst, 2) if worst is not None else None,
        waiting_for_speaker=waiting and chunks == 0,
    )
