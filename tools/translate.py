#!/usr/bin/env python3
"""
Translate text directly, with no speech recognition in the way.

Judging translation quality through a live microphone conflates two things.
Whisper mishears, the translator faithfully renders the mishearing, and the
output looks like a translation failure when it is a transcription failure.
This takes text straight to the translation backend so the quality being
judged is the translator's.

    python tools/translate.py --to te "Good morning everyone and welcome"
    python tools/translate.py --to te,hi,fr --file passage.txt
    python tools/translate.py --to te --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.adapters.mt_routing import RoutingMtAdapter
from lad_translate.api.languages import describe
from lad_translate.obs.log import configure


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="text to translate")
    ap.add_argument("--to", default="te", help="comma separated target languages")
    ap.add_argument("--file", type=Path, help="translate each line of a file")
    ap.add_argument("--interactive", action="store_true", help="type lines, see translations")
    args = ap.parse_args()

    configure("ERROR")
    targets = [t.strip() for t in args.to.split(",") if t.strip()]

    mt = RoutingMtAdapter(
        "en", targets,
        opus_options={"model_root": ROOT / "models" / "mt"},
        nllb_options={"model_path": ROOT / "models" / "mt" / "nllb-600m"},
    )
    for code in targets:
        info = describe(code)
        print(f"  {info.native} ({info.english}) via {mt.backend_for(code)}")
    print()

    async def render(line: str) -> None:
        line = line.strip()
        if not line:
            return
        out = await mt.translate_many(line, "en", targets)
        print(f"  EN  {line}")
        for code in targets:
            print(f"  {code.upper()}  {out.get(code, '')}")
        print()

    try:
        if args.file:
            for line in args.file.read_text().splitlines():
                await render(line)
        elif args.interactive:
            print("  Type a sentence and press enter. Ctrl-D to finish.\n")
            loop = asyncio.get_running_loop()
            while True:
                try:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    break
                await render(line)
        elif args.text:
            await render(" ".join(args.text))
        else:
            ap.error("give text, --file, or --interactive")
    finally:
        mt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
