#!/usr/bin/env python3
"""
Word error rate against a ground truth transcript.

WER is the standard measure: substitutions plus deletions plus insertions,
over the number of reference words. 0% is perfect, and it can exceed 100% when
a model hallucinates more words than were spoken.

The reference must come from the published text, not from a larger model's
output. Scoring one model against another measures agreement, not accuracy,
and both can be confidently wrong in the same place.

Usage:
    python tools/score_stt.py --audio fixtures/holmes.wav --reference fixtures/holmes.txt
    python tools/score_stt.py --audio fixtures/holmes.wav --models tiny,base,small
    python tools/score_stt.py --reference fixtures/holmes.txt --session <uuid>
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Casing and punctuation are not what STT is being judged on here, and Whisper
# capitalises and punctuates by its own rules. Numerals are the exception worth
# noting: "eleven" and "11" score as an error, which is correct for a
# translation pipeline because the two tokenise differently downstream.
_PUNCT = re.compile(r"[^\w\s']")


def normalise(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def wer(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """
    Levenshtein over words. Returns (substitutions, deletions, insertions, distance).

    Full DP table rather than a rolling row: the counts of each edit type are
    worth having separately. A model that drops half the audio and one that
    invents words both score badly, and they are different problems.
    """
    n, m = len(reference), len(hypothesis)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dist[i][0], back[i][0] = i, "d"
    for j in range(1, m + 1):
        dist[0][j], back[0][j] = j, "i"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dist[i][j], back[i][j] = dist[i - 1][j - 1], "="
                continue
            sub, dele, ins = dist[i - 1][j - 1] + 1, dist[i - 1][j] + 1, dist[i][j - 1] + 1
            best = min(sub, dele, ins)
            dist[i][j] = best
            back[i][j] = "s" if best == sub else ("d" if best == dele else "i")

    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "=":
            i, j = i - 1, j - 1
        elif op == "s":
            subs += 1
            i, j = i - 1, j - 1
        elif op == "d":
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1
    return subs, dels, inss, dist[n][m]


def report(label: str, reference: list[str], hypothesis: list[str], extra: str = "") -> float:
    subs, dels, inss, distance = wer(reference, hypothesis)
    rate = distance / len(reference) if reference else 0.0
    print(
        f"  {label:<22} WER {rate * 100:6.1f}%   "
        f"sub {subs:3d}  del {dels:3d}  ins {inss:3d}   "
        f"{len(hypothesis):3d}/{len(reference)} words{extra}"
    )
    return rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=ROOT / "fixtures" / "holmes.txt")
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "holmes.wav")
    ap.add_argument("--models", default="tiny,base,small", help="batch models to score")
    ap.add_argument("--session", help="also score what a live session actually produced")
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"error: no reference at {args.reference}", file=sys.stderr)
        return 1
    reference = normalise(args.reference.read_text())
    print(f"\nreference: {args.reference.name}, {len(reference)} words\n")

    if not args.no_batch:
        from faster_whisper import WhisperModel

        print("batch (whole file at once):")
        for size in [m.strip() for m in args.models.split(",") if m.strip()]:
            started = time.monotonic()
            model = WhisperModel(size, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(args.audio), language="en", beam_size=1)
            text = " ".join(s.text.strip() for s in segments)
            elapsed = time.monotonic() - started
            report(size, reference, normalise(text), extra=f"   [{elapsed:.0f}s]")
            del model
        print()

    if args.session:
        import asyncio
        import os

        async def live() -> None:
            import asyncpg

            url = os.environ.get(
                "LAD_DATABASE_URL", "postgresql://lad@127.0.0.1:55432/salesmaya_agent"
            )
            pool = await asyncpg.create_pool(url, min_size=1, max_size=2)
            try:
                schema = await pool.fetchval(
                    "SELECT schema_name FROM lad_dev.tenants WHERE slug='techiemaya'"
                )
                rows = await pool.fetch(
                    f"""SELECT DISTINCT ON (chunk_id) chunk_id, source_text
                        FROM {schema}.session_transcripts
                        WHERE session_id = $1::uuid ORDER BY chunk_id""",
                    args.session,
                )
            finally:
                await pool.close()
            if not rows:
                print(f"no transcript rows for session {args.session}")
                return

            # A looping demo replays the clip, so the session holds several
            # passes over a single-pass reference. Scoring all of them against
            # one reports every repeat as an insertion: an early run of this
            # gave 298% WER with 432 insertions, which said nothing about the
            # pipeline and everything about the measurement. Score one pass.
            words: list[str] = []
            used = 0
            for row in rows:
                chunk = normalise(row["source_text"])
                if words and len(words) + len(chunk) > len(reference) * 1.25:
                    break
                words.extend(chunk)
                used += 1

            looped = used < len(rows)
            print("live (streaming through the pipeline):")
            report(
                "live session", reference, words,
                extra=f"   [{used} of {len(rows)} chunks"
                      + (", first pass only" if looped else "") + "]",
            )
            if looped:
                print(
                    f"       source looped: {len(rows)} chunks recorded, scored the "
                    f"first {used} that cover one pass"
                )
            print()

        asyncio.run(live())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
