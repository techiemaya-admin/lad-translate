"""
Read and write /etc/lad-translate/session.env.

Rewritten in place, preserving comments and order, because that file carries
the reasoning behind every value - what was measured, what it cost, why a
number is what it is. A console that regenerated it from a template would throw
that away on the first save, and the comments are most of what stops someone
setting emit 1.0 at a venue.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_PATH = Path("/etc/lad-translate/session.env")

_ASSIGN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")

# Only these may be set from the console. A allowlist rather than a denylist:
# this file also holds the database URL's schema and the LiveKit addresses, and
# a console that can rewrite those can point a venue at the wrong SFU.
EDITABLE = frozenset(
    {
        "LAD_TRANSLATE_TARGETS",
        "LAD_TRANSLATE_EVENT",
        "LAD_TRANSLATE_STT_MODEL",
        "LAD_TRANSLATE_STT_DEVICE",
        "LAD_TRANSLATE_STT_THREADS",
        "LAD_TRANSLATE_LOOKAHEAD",
        "LAD_TRANSLATE_VAD_FLAG",
        "LAD_TRANSLATE_EMIT_INTERVAL",
        "LAD_TRANSLATE_WINDOW",
        "LAD_TRANSLATE_WAIT",
        "STT_BACKEND",
    }
)


def read(path: Path = DEFAULT_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGN.match(stripped)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def write(updates: dict[str, str], path: Path = DEFAULT_PATH) -> list[str]:
    """
    Apply `updates` in place. Returns the keys that actually changed.

    Rejects anything outside EDITABLE rather than silently skipping it, so a
    typo in a key name is a visible error instead of a setting that appears to
    save and does not.
    """
    unknown = sorted(set(updates) - EDITABLE)
    if unknown:
        raise ValueError(f"not editable from the console: {', '.join(unknown)}")

    lines = path.read_text().splitlines()
    changed: list[str] = []
    seen: set[str] = set()

    for i, line in enumerate(lines):
        match = _ASSIGN.match(line.strip())
        if not match:
            continue
        key = match.group(1)
        if key not in updates:
            continue
        seen.add(key)
        new = str(updates[key])
        if match.group(2) != new:
            lines[i] = f"{key}={new}"
            changed.append(key)

    # A key the file has never carried is appended rather than dropped. Same
    # reasoning as bootstrap.sh: a release that adds a setting must reach a box
    # that predates it, or the unit references a variable that expands to
    # nothing and the service dies in argparse.
    for key in updates:
        if key not in seen:
            lines.append(f"{key}={updates[key]}")
            changed.append(key)

    if changed:
        path.write_text("\n".join(lines) + "\n")
    return changed
