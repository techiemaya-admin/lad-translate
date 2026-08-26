#!/usr/bin/env python3
"""
Offline chunker tuning harness.

Answers the question the architecture depends on: how long must the chunker
hold a transcript before committing it, and what does committing earlier cost
in words the audience hears wrongly.

Runs with no GPU, no network and no STT model, so it is usable on the dev Mac.

Two sources of hypotheses:

    --source sim     a model of streaming STT revision behaviour. Useful for
                     sweeping the parameter space. The absolute numbers are
                     only as good as the model's assumptions.

    --source jsonl   hypotheses recorded from a real backend, one JSON object
                     per line. This is the only source whose numbers may be
                     quoted. Capture them with tools/record_hypotheses.py once
                     an STT backend is installed.

Usage:
    python tools/chunker_replay.py --sweep
    python tools/chunker_replay.py --agreement 2 --max-wait 1.0 --verbose
    python tools/chunker_replay.py --source jsonl --jsonl fixtures/keynote.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lad_translate.adapters.base import Hypothesis  # noqa: E402
from lad_translate.chunker import ChunkerConfig, CommitReason, PhraseChunker  # noqa: E402

# =============================================================================
# SAMPLE SOURCE TEXT
# =============================================================================

KEYNOTE = """
Good morning everyone, and thank you for joining us today.
Over the next forty minutes I want to walk through three things.
First, where the market actually stands right now, because a lot of the
reporting has been misleading.
Second, what we learned from the pilot programme in Sharjah, which did not go
the way we expected.
And third, what that means for the roadmap over the next eighteen months.
Let me start with the market.
Revenue across the sector grew eleven percent last year, but that number hides
an enormous amount of variation.
The top four operators took almost all of that growth.
Everyone else was flat or declining.
So when someone tells you the sector is healthy, ask them which part.
Now, the pilot.
We ran it across six venues between March and August.
The technology worked. The logistics did not.
"""

# =============================================================================
# SIMULATED BACKEND
# =============================================================================


@dataclass(slots=True)
class BackendProfile:
    """
    How a streaming STT backend behaves, in the two ways the chunker cares about.

    Defaults describe a Deepgram-class streaming backend. For a Whisper
    sliding-window wrapper, expect emit_lag around 1.0 to 1.5 and an
    unstable_tail of 5 or more, which is why streaming Whisper struggles here.
    """

    name: str = "deepgram-class"
    interval: float = 0.15
    """Wall seconds between interim emissions."""

    emit_lag: float = 0.25
    """Wall delay between audio being spoken and appearing in a hypothesis."""

    unstable_tail: int = 3
    """Trailing words the backend may still revise."""

    error_rate: float = 0.25
    """Chance a word in the unstable tail is currently wrong."""

    words_per_second: float = 2.8
    """Speaking rate, about 170 words per minute."""


WHISPER_STREAMING = BackendProfile(
    name="whisper-streaming-class",
    interval=0.5,
    emit_lag=1.2,
    unstable_tail=6,
    error_rate=0.35,
)


def simulate(text: str, profile: BackendProfile, seed: int = 7) -> tuple[list[Hypothesis], list[str]]:
    """
    Produce a hypothesis stream plus the ground truth token list.

    Words outside the unstable tail are locked to truth. Words inside it are
    re-drawn on every emission, so they churn exactly the way real interims do.
    """
    rng = random.Random(seed)
    truth = text.split()
    hyps: list[Hypothesis] = []

    wall = 0.0
    seq = 0
    while True:
        # How much audio the backend has processed by now.
        heard = (wall - profile.emit_lag) * profile.words_per_second
        n = int(max(0.0, heard))
        if n <= 0:
            wall += profile.interval
            continue
        n = min(n, len(truth))

        words = list(truth[:n])
        tail_start = max(0, n - profile.unstable_tail)
        for i in range(tail_start, n):
            if rng.random() < profile.error_rate:
                words[i] = _corrupt(words[i], rng)

        is_final = n >= len(truth)
        hyps.append(
            Hypothesis(
                text=" ".join(words),
                is_final=is_final,
                t_audio_start=0.0,
                t_audio_end=n / profile.words_per_second,
                t_wall=wall,
                seq=seq,
            )
        )
        seq += 1
        if is_final:
            break
        wall += profile.interval

    return hyps, truth


def _corrupt(word: str, rng: random.Random) -> str:
    """Mangle a word the way a backend mis-hears one, keeping punctuation."""
    head = word.rstrip(".,;:!?")
    tail = word[len(head) :]
    if len(head) < 3:
        return word
    cut = rng.randrange(1, len(head))
    return head[:cut] + rng.choice("aeiourstn") + head[cut + 1 :] + tail


def load_jsonl(path: Path) -> tuple[list[Hypothesis], list[str]]:
    """Load hypotheses recorded from a real backend."""
    hyps: list[Hypothesis] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            hyps.append(
                Hypothesis(
                    text=d["text"],
                    is_final=d["is_final"],
                    t_audio_start=d["t_audio_start"],
                    t_audio_end=d["t_audio_end"],
                    t_wall=d["t_wall"],
                    seq=d["seq"],
                )
            )
    truth = hyps[-1].text.split() if hyps else []
    return hyps, truth

# =============================================================================
# MEASUREMENT
# =============================================================================


@dataclass(slots=True)
class Result:
    label: str
    chunks: int
    mean_words: float
    p50_commit_latency: float
    p95_commit_latency: float
    max_commit_latency: float
    wrong_word_rate: float
    reasons: dict[str, int]

    def row(self) -> str:
        top = max(self.reasons, key=lambda k: self.reasons[k]) if self.reasons else "-"
        return (
            f"{self.label:<26} {self.chunks:>6} {self.mean_words:>7.1f} "
            f"{self.p50_commit_latency:>8.2f} {self.p95_commit_latency:>8.2f} "
            f"{self.max_commit_latency:>8.2f} {self.wrong_word_rate * 100:>8.1f}%  {top}"
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


def evaluate(
    hyps: list[Hypothesis], truth: list[str], config: ChunkerConfig, label: str
) -> tuple[Result, list]:
    chunker = PhraseChunker(config)
    chunks = []
    for hyp in hyps:
        chunks.extend(chunker.feed(hyp))
    chunks.extend(chunker.flush())

    # Commit latency: speaker finishes the phrase, chunker releases it.
    # Audio position t maps to wall time t, so this includes backend lag AND
    # the chunker's own hold. It is the chunker stage of the latency budget.
    latencies = [max(0.0, c.t_wall_committed - c.t_audio_end) for c in chunks]

    committed = " ".join(c.text for c in chunks).split()
    wrong = sum(1 for a, b in zip(committed, truth) if a != b)
    wrong += abs(len(committed) - len(truth))
    rate = wrong / len(truth) if truth else 0.0

    reasons: dict[str, int] = {}
    for c in chunks:
        reasons[c.reason.value] = reasons.get(c.reason.value, 0) + 1

    return (
        Result(
            label=label,
            chunks=len(chunks),
            mean_words=statistics.fmean([c.word_count for c in chunks]) if chunks else 0.0,
            p50_commit_latency=percentile(latencies, 0.50),
            p95_commit_latency=percentile(latencies, 0.95),
            max_commit_latency=max(latencies) if latencies else 0.0,
            wrong_word_rate=rate,
            reasons=reasons,
        ),
        chunks,
    )


HEADER = (
    f"{'config':<26} {'chunks':>6} {'words':>7} {'p50':>8} {'p95':>8} "
    f"{'max':>8} {'wrong':>9}  top reason"
)

# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sim", "jsonl"], default="sim")
    ap.add_argument("--jsonl", type=Path)
    ap.add_argument("--profile", choices=["deepgram", "whisper"], default="deepgram")
    ap.add_argument("--sweep", action="store_true", help="sweep agreement_n and max_wait_s")
    ap.add_argument("--agreement", type=int, default=2)
    ap.add_argument("--max-wait", type=float, default=1.0)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="print every chunk")
    args = ap.parse_args()

    if args.source == "jsonl":
        if not args.jsonl or not args.jsonl.exists():
            print("error: --jsonl path is required and must exist", file=sys.stderr)
            return 2
        hyps, truth = load_jsonl(args.jsonl)
        source_label = f"recorded: {args.jsonl.name}"
        credible = True
    else:
        profile = WHISPER_STREAMING if args.profile == "whisper" else BackendProfile()
        text = " ".join(KEYNOTE.split())
        hyps, truth = simulate(text, profile, seed=args.seed)
        source_label = f"simulated: {profile.name}"
        credible = False

    print(f"\nsource      {source_label}")
    print(f"hypotheses  {len(hyps)}")
    print(f"truth       {len(truth)} words\n")

    if args.sweep:
        print(HEADER)
        print("-" * len(HEADER))
        for agreement in (1, 2, 3):
            for wait in (0.3, 0.6, 1.0, 1.5):
                cfg = ChunkerConfig(
                    agreement_n=agreement,
                    max_wait_s=wait,
                    min_words=args.min_words,
                    max_words=args.max_words,
                )
                result, _ = evaluate(hyps, truth, cfg, f"agreement={agreement} wait={wait}")
                print(result.row())
    else:
        cfg = ChunkerConfig(
            agreement_n=args.agreement,
            max_wait_s=args.max_wait,
            min_words=args.min_words,
            max_words=args.max_words,
        )
        result, chunks = evaluate(hyps, truth, cfg, f"agreement={args.agreement} wait={args.max_wait}")
        print(HEADER)
        print("-" * len(HEADER))
        print(result.row())
        if args.verbose:
            print("\nchunks:")
            for c in chunks:
                flag = "  REVISED" if c.revised_after_commit else ""
                print(f"  [{c.chunk_id:>3}] {c.reason.value:<18} lag={c.stability_lag:.2f}s  {c.text}{flag}")

    if not credible:
        print(
            "\nNOTE  These figures come from a behaviour model, not a real backend.\n"
            "      Use them to compare configurations against each other. Do not\n"
            "      quote them as product latency. Record real hypotheses first."
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
