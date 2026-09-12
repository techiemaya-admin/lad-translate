"""
Recordings on disk, as the console lists, serves and deletes them.

Layout, written by session/recording.py:

    <root>/<room>/<session_id>-<stamp>/{manifest.json, source.wav, <lang>.wav}

Two rules that keep a download endpoint from becoming a file server:

  - A take directory is identified by its name, which must look like one
    (a UUID, a dash, a timestamp). Anything else is refused before it
    touches the filesystem. Path traversal is a shape mismatch, not a
    special case.
  - A file within a take is served only if it is one the recorder writes:
    manifest.json, source.wav, or a short language code dot wav.

Deletion removes one take, never a room and never the root.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .sessions import validate_room

TAKE_NAME = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-\d{8}T\d{6}Z$")
FILE_NAME = re.compile(r"^(manifest\.json|source\.wav|[a-z]{2,3}\.wav)$")


class BadName(ValueError):
    pass


@dataclass(frozen=True)
class Take:
    room: str
    name: str
    session_id: str
    started_at: str | None
    ended_at: str | None
    seconds: float
    bytes: int
    files: list[dict]
    in_progress: bool

    def as_dict(self) -> dict:
        return asdict(self)


def validate_take(name: str) -> str:
    if not TAKE_NAME.match(name):
        raise BadName(f"{name!r} is not a recording name")
    return name


def validate_file(name: str) -> str:
    if not FILE_NAME.match(name):
        raise BadName(f"{name!r} is not a recording file")
    return name


def _describe(room: str, directory: Path) -> Take:
    manifest: dict = {}
    with contextlib.suppress(OSError, ValueError):
        manifest = json.loads((directory / "manifest.json").read_text())
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and FILE_NAME.match(path.name):
            files.append({"name": path.name, "bytes": path.stat().st_size})
    return Take(
        room=room,
        name=directory.name,
        session_id=manifest.get("session_id") or directory.name.split("-2")[0],
        started_at=manifest.get("started_at"),
        ended_at=manifest.get("ended_at"),
        seconds=float(manifest.get("seconds") or 0.0),
        bytes=sum(f["bytes"] for f in files),
        files=files,
        in_progress=bool(manifest) and manifest.get("ended_at") is None,
    )


def list_takes(root: Path, room: str) -> list[Take]:
    validate_room(room)
    base = root / room
    if not base.is_dir():
        return []
    takes = [_describe(room, d) for d in base.iterdir() if d.is_dir() and TAKE_NAME.match(d.name)]
    return sorted(takes, key=lambda t: t.name, reverse=True)


def take_dir(root: Path, room: str, name: str) -> Path:
    validate_room(room)
    validate_take(name)
    path = root / room / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    return path


def file_path(root: Path, room: str, name: str, filename: str) -> Path:
    validate_file(filename)
    path = take_dir(root, room, name) / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


def delete_take(root: Path, room: str, name: str) -> None:
    path = take_dir(root, room, name)
    shutil.rmtree(path)


def disk(root: Path) -> dict:
    """Free space where the recordings live, for the console to show."""
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        return {"free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        return {"free_bytes": None, "total_bytes": None}
