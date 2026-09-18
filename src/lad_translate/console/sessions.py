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
import shutil
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
    recording: bool = False
    recording_dir: str | None = None
    recording_takes: int = 0


class NotManagedHere(RuntimeError):
    """
    This host does not run the session units, so the deck cannot drive them.

    Distinct from a unit that failed: nothing is broken, the console is simply
    somewhere systemd is not. Raised BEFORE shelling out, because the errors
    you get otherwise are misleading - on macOS `sudo` exists while systemctl
    does not, so `sudo -n systemctl restart` answers "sudo: a password is
    required" and sends the operator hunting for a sudoers rule that would not
    have helped.
    """


def _explain_if_no_systemd() -> None:
    """
    Call ONLY after a command has failed, never before it.

    Checking up front runs ahead of a patched _run, so the suite would pass on
    a Linux box and fail on a Mac purely from the host it ran on. Tests that
    stub the command out stay on the happy path and never reach here.
    """
    if shutil.which("systemctl") is None:
        raise NotManagedHere(
            "this host has no systemd, so the console cannot start or stop "
            "sessions here - run tools/session_live.py from a terminal instead"
        )


async def _run(*args: str) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except FileNotFoundError:
        # No systemd on this host. The deck cannot manage units here, but the
        # rest of the console - transcript, recordings, outputs, QR - reads the
        # database and files and works perfectly well, so report the unit as
        # absent rather than 500 the whole page.
        #
        # Found running the console on a Mac to test locally: status() 500'd on
        # FileNotFoundError from systemctl and took the deck down with it. On
        # the VM the binary is always there, which is why this never showed.
        return 127, (
            f"{args[0]} is not installed on this host, so the console cannot "
            "manage session units here - run tools/session_live.py instead"
        )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


async def restart(room: str) -> None:
    validate_room(room)
    code, out = await _run("sudo", "-n", "systemctl", "restart", UNIT.format(room=room))
    if code != 0:
        _explain_if_no_systemd()
        raise RuntimeError(f"systemctl restart failed: {out.strip()[:300]}")
    log.info("session restarted", extra={"room": room})


async def stop(room: str) -> None:
    validate_room(room)
    code, out = await _run("sudo", "-n", "systemctl", "stop", UNIT.format(room=room))
    if code != 0:
        _explain_if_no_systemd()
        raise RuntimeError(f"systemctl stop failed: {out.strip()[:300]}")
    log.info("session stopped", extra={"room": room})


async def record(room: str, on: bool) -> None:
    """
    Tell a running session to start or stop recording.

    SIGUSR1 starts, SIGUSR2 stops - both idempotent in the session, so this
    can say "be recording" without first asking whether it is. A session that
    is not running is not an error here: the caller has already written the
    env flag, so the next start will honour it.
    """
    validate_room(room)
    sig = "SIGUSR1" if on else "SIGUSR2"
    code, out = await _run(
        "sudo", "-n", "systemctl", "kill", f"--signal={sig}", UNIT.format(room=room)
    )
    if code != 0 and "not loaded" not in out and "inactive" not in out:
        _explain_if_no_systemd()
        raise RuntimeError(f"systemctl kill failed: {out.strip()[:300]}")
    log.info("recording signalled", extra={"room": room, "on": on})


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
    recording = False
    recording_dir = None
    takes = 0

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
        elif message == "recording started":
            recording, recording_dir = True, entry.get("directory")
            takes = int(entry.get("take", takes + 1))
        elif message == "recording stopped":
            recording = False
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
        recording=recording and active,
        recording_dir=recording_dir if (recording and active) else None,
        recording_takes=takes,
    )
