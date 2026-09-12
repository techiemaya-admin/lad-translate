"""
The venue output agent's two regressions, both found at the user's desk.

Neither needed a model, a room or a sound card to reproduce once it was
understood, which is the point of pinning them here: they were found by an
operator hearing something wrong, and they should never be found that way
twice.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from output_agent import SingleDrain  # noqa: E402

UNIT = ROOT / "deploy" / "vm" / "lad-translate-session@.service"


# --- one stream per language ------------------------------------------------


@pytest.mark.asyncio
async def test_replacing_cancels_the_previous_stream():
    """
    The bug: a session restart republishes the language tracks, the room
    re-subscribes, and the handler started ANOTHER drain into the same ring.
    Five restarts produced five copies of every phrase - audio that keeps
    going after the speaker stops.
    """
    pushed: list[str] = []

    async def stream(tag: str) -> None:
        try:
            while True:
                pushed.append(tag)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    drain = SingleDrain()
    drain.replace(stream("first"))
    await asyncio.sleep(0.05)
    drain.replace(stream("second"))
    await asyncio.sleep(0.05)

    assert drain.replaced == 1
    tail = pushed[-4:]
    assert set(tail) == {"second"}, f"the first stream is still pushing: {tail}"
    await drain.stop()


@pytest.mark.asyncio
async def test_five_restarts_leave_one_stream_running():
    """The shape of what actually happened, five times over."""
    running = 0

    async def stream() -> None:
        nonlocal running
        running += 1
        try:
            await asyncio.sleep(3600)
        finally:
            running -= 1

    drain = SingleDrain()
    for _ in range(5):
        drain.replace(stream())
        await asyncio.sleep(0.01)
    assert running == 1, f"{running} streams into one ring"
    assert drain.replaced == 4
    await drain.stop()
    await asyncio.sleep(0.01)
    assert running == 0


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_leaves_nothing_running():
    drain = SingleDrain()
    await drain.stop()          # nothing started yet
    drain.replace(asyncio.sleep(3600))
    await drain.stop()
    await drain.stop()
    assert not drain.running


@pytest.mark.asyncio
async def test_a_finished_stream_is_not_counted_as_replaced():
    """A track that ended on its own is not a stream we had to cancel."""
    drain = SingleDrain()
    drain.replace(asyncio.sleep(0))
    await asyncio.sleep(0.02)
    drain.replace(asyncio.sleep(3600))
    assert drain.replaced == 0
    await drain.stop()


# --- the unit file's optional flags -----------------------------------------


def test_optional_flags_use_dollar_var_not_braces():
    """
    systemd substitutes ${VAR} as exactly ONE argument, so an empty value
    becomes an empty-string argument and argparse exits 2 with
    "unrecognized arguments:" and nothing after it. $VAR word-splits, and an
    empty value produces no argument at all.

    LAD_TRANSLATE_RECORD_FLAG is empty whenever recording is off, so the
    braced form crash-looped the session the moment the recording release
    landed - on a box where the unit tests, the linter and the bootstrap
    were all green, because none of them runs the unit.
    """
    text = UNIT.read_text()
    optional = [line.strip() for line in text.splitlines()
                if "_FLAG" in line and line.strip().startswith(("$", "${"))]
    assert optional, "expected the optional flags to be on their own lines"
    for line in optional:
        assert not line.startswith("${"), (
            f"{line!r} uses ${{VAR}}; an empty value becomes an empty argument. Use $VAR."
        )


def test_every_env_reference_in_execstart_is_defined_in_the_example():
    """
    A unit that references a variable session.env has never heard of gets an
    empty expansion, and bootstrap's key-appending only adds keys the example
    file carries. The two have to agree.
    """
    unit = UNIT.read_text()
    example = (ROOT / "deploy" / "vm" / "session.env.example").read_text()
    defined = set(re.findall(r"^([A-Z_]+)=", example, re.M))
    # Only the ExecStart line block references env; LIVEKIT_* come from
    # secrets.env, which is not in the example.
    referenced = set(re.findall(r"\$\{?(LAD_TRANSLATE_[A-Z_]+)\}?", unit))
    missing = sorted(referenced - defined)
    assert not missing, f"{missing} used by the unit but absent from session.env.example"
