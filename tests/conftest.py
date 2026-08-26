"""Shared helpers for building synthetic hypothesis streams."""

from __future__ import annotations

from collections.abc import Iterator

from lad_translate.adapters.base import Hypothesis

WORD_SECONDS = 0.35
"""Rough conference speaking rate, about 170 words per minute."""

HYPOTHESIS_INTERVAL = 0.15
"""How often a streaming backend emits an interim, in wall seconds."""


def stream(
    stages: list[str],
    *,
    start_wall: float = 100.0,
    final: bool = True,
) -> Iterator[Hypothesis]:
    """
    Build hypotheses from explicit transcript stages.

    Each stage is the backend's full transcript at that moment. Stages may
    shrink or contradict each other, which is exactly what real backends do.
    """
    wall = start_wall
    for i, text in enumerate(stages):
        words = text.split()
        is_final = final and i == len(stages) - 1
        yield Hypothesis(
            text=text,
            is_final=is_final,
            t_audio_start=0.0,
            t_audio_end=len(words) * WORD_SECONDS,
            t_wall=wall,
            seq=i,
        )
        wall += HYPOTHESIS_INTERVAL


def growing(sentence: str, *, start_wall: float = 100.0, final: bool = True) -> Iterator[Hypothesis]:
    """A clean backend that only ever appends, one word at a time."""
    words = sentence.split()
    stages = [" ".join(words[: i + 1]) for i in range(len(words))]
    return stream(stages, start_wall=start_wall, final=final)
