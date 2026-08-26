#!/usr/bin/env python3
"""
Fetch Piper voices.

Voices come from rhasspy/piper-voices, the project's own repository, so the
provenance problem that affects the translation models does not apply here.

Voice choice is a product decision. The audience hears these through cheap
earbuds at conference volume, so pick for clarity rather than warmth, and
listen to a candidate in a noisy room before committing it to an event.

Usage:
    python tools/fetch_tts_voices.py --defaults
    python tools/fetch_tts_voices.py --voice fr_FR-tom-medium
    python tools/fetch_tts_voices.py --list-available fr
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

VOICE_ROOT = Path(__file__).resolve().parent.parent / "models" / "tts"
HUB_API = "https://huggingface.co/api/models/rhasspy/piper-voices"

DEFAULTS = [
    "fr_FR-siwis-medium",
    "de_DE-thorsten-medium",
    "es_ES-davefx-medium",
    "ar_JO-kareem-medium",
    "te_IN-maya-medium",
    "hi_IN-pratham-medium",
]


def available(language: str) -> list[str]:
    with urllib.request.urlopen(HUB_API, timeout=30) as response:
        payload = json.load(response)
    return sorted(
        f["rfilename"].rsplit("/", 1)[-1][:-5]
        for f in payload.get("siblings", [])
        if f["rfilename"].endswith(".onnx") and f["rfilename"].startswith(f"{language}/")
    )


def fetch(voice: str, force: bool) -> int:
    from piper.download_voices import download_voice

    model = VOICE_ROOT / f"{voice}.onnx"
    if model.exists() and not force:
        print(f"{voice:<26} already present")
        return 0
    try:
        download_voice(voice, VOICE_ROOT)
    except Exception as exc:  # noqa: BLE001 - the operator needs the real cause
        print(f"{voice:<26} FAILED: {exc}", file=sys.stderr)
        return 1
    size = model.stat().st_size / 1e6 if model.exists() else 0
    print(f"{voice:<26} ready, {size:.0f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--voice", action="append", default=[])
    ap.add_argument("--defaults", action="store_true", help="fetch the configured set")
    ap.add_argument("--list-available", metavar="LANG", help="list voices for a language code")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list_available:
        for name in available(args.list_available):
            print(" ", name)
        return 0

    voices = list(args.voice)
    if args.defaults:
        voices.extend(DEFAULTS)
    if not voices:
        ap.error("give --voice, --defaults, or --list-available")

    VOICE_ROOT.mkdir(parents=True, exist_ok=True)
    return 1 if sum(fetch(v, args.force) for v in dict.fromkeys(voices)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
