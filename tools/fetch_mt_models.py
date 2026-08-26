#!/usr/bin/env python3
"""
Fetch Opus-MT translation models in CTranslate2 format.

Two routes, and the difference matters.

VERIFIED route (default). Downloads a pre-converted CTranslate2 build from the
Hub. Fast, no torch needed, works on the dev Mac. But these repos belong to
unaffiliated individuals, coverage is patchy, and several advertise a model
while shipping no weights at all. Every entry in VERIFIED below was checked to
contain both model.bin and source.spm on 2026-08-25.

CONVERT route (--convert). Converts the official Helsinki-NLP model with
ctranslate2's own converter. Needs transformers and torch, so it will not run
on the dev Mac, but it is the only route fit for production: the source is
official, the conversion is reproducible, and the output should be stored in
our own bucket rather than pulled from the Hub at deploy time.

Do not ship a venue on the VERIFIED route. A third party can delete or alter
those repos at any time, and the failure lands mid-event.

Usage:
    python tools/fetch_mt_models.py --pair en-fr
    python tools/fetch_mt_models.py --pair en-fr --pair en-de --pair en-es
    python tools/fetch_mt_models.py --list
    python tools/fetch_mt_models.py --pair en-ar --convert    # needs torch
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "mt"

# Pre-converted CTranslate2 builds, each verified to carry real weights.
VERIFIED: dict[str, str] = {
    "en-fr": "michaelfeil/ct2fast-opus-mt-en-fr",
    "en-de": "michaelfeil/ct2fast-opus-mt-en-de",
    "en-es": "michaelfeil/ct2fast-opus-mt-en-es",
    "en-zh": "gaudi/opus-mt-en-zh-ctranslate2",
    "en-ar": "ooeoeo/opus-mt-en-ar-ct2-float16",
    # Dedicated pair model, so no target token is needed. Preferred over the
    # en-inc family model: one pair beats capacity split across a family.
    "en-hi": "manancode/opus-mt-en-hi-ctranslate2-android",
    # Family model covering Telugu, Tamil, Malayalam and Kannada. There is no
    # dedicated en-te model. Needs a sentence-initial token (>>tel<< and so on)
    # to pick the target: see MULTILINGUAL in adapters/mt_opus.py. Without it
    # the model translates into whichever family member it likes, silently.
    "en-dra": "manancode/opus-mt-en-dra-ctranslate2-android",
}

# Official sources for the convert route. Complete where the pre-converted
# ecosystem is not.
OFFICIAL_TEMPLATE = "Helsinki-NLP/opus-mt-{source}-{target}"

NLLB_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
"""One model, 200 languages, already int8. See adapters/mt_nllb.py."""

REQUIRED_FILES = ("model.bin", "source.spm", "target.spm", "config.json")


def verify(path: Path) -> list[str]:
    """Return the list of required files that are missing."""
    return [f for f in REQUIRED_FILES if not (path / f).exists()]


def fetch_verified(pair: str, force: bool) -> int:
    if pair not in VERIFIED:
        print(f"error: no verified build for {pair}.", file=sys.stderr)
        print(f"       Available: {', '.join(sorted(VERIFIED))}", file=sys.stderr)
        print(f"       Or convert the official model: --pair {pair} --convert", file=sys.stderr)
        return 1

    from huggingface_hub import snapshot_download

    dest = MODEL_ROOT / pair
    if dest.exists() and not verify(dest) and not force:
        print(f"{pair:<8} already present at {dest}")
        return 0
    if force and dest.exists():
        shutil.rmtree(dest)

    repo = VERIFIED[pair]
    print(f"{pair:<8} downloading {repo}")
    snapshot_download(repo, local_dir=str(dest))

    missing = verify(dest)
    if missing:
        print(f"error: {repo} is missing {missing}", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"{pair:<8} ready, {size:.0f} MB  (source: {repo}, third party)")
    return 0


def fetch_converted(pair: str, quantization: str) -> int:
    """Convert the official Helsinki-NLP model. Needs transformers and torch."""
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print("error: ctranslate2 converters unavailable", file=sys.stderr)
        return 1

    source, _, target = pair.partition("-")
    repo = OFFICIAL_TEMPLATE.format(source=source, target=target)
    dest = MODEL_ROOT / pair
    print(f"{pair:<8} converting {repo} (quantization={quantization})")

    try:
        converter = TransformersConverter(repo)
        converter.convert(str(dest), quantization=quantization, force=True)
    except Exception as exc:
        print(f"error: conversion failed: {exc}", file=sys.stderr)
        print("       This route needs transformers and torch installed.", file=sys.stderr)
        print("       It will not run on an Intel Mac; use the GPU box.", file=sys.stderr)
        return 1

    missing = verify(dest)
    if missing:
        print(f"error: conversion produced no {missing}", file=sys.stderr)
        return 1
    print(f"{pair:<8} ready  (source: {repo}, official)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pair", action="append", default=[], metavar="SRC-TGT")
    ap.add_argument("--convert", action="store_true", help="convert the official model instead")
    ap.add_argument("--quantization", default="int8", choices=["int8", "int8_float16", "float16", "float32"])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--list", action="store_true", help="show verified builds and local state")
    ap.add_argument("--nllb", action="store_true",
                    help="fetch NLLB-200 distilled 600M (one model, 200 languages, GPU backend)")
    args = ap.parse_args()

    if args.list:
        print(f"\nmodel root: {MODEL_ROOT}\n")
        print(f"{'pair':<10} {'local':<10} source")
        print("-" * 64)
        for pair, repo in sorted(VERIFIED.items()):
            path = MODEL_ROOT / pair
            state = "missing" if not path.exists() else ("ok" if not verify(path) else "incomplete")
            print(f"{pair:<10} {state:<10} {repo}")
        print(
            "\nAll of the above are third party conversions. For production, run\n"
            "with --convert on a machine with torch and store the output in our\n"
            "own bucket.\n"
        )
        return 0

    if args.nllb:
        from huggingface_hub import snapshot_download

        dest = MODEL_ROOT / "nllb-600m"
        print(f"nllb     downloading {NLLB_REPO}")
        snapshot_download(NLLB_REPO, local_dir=str(dest))
        if not (dest / "model.bin").exists():
            print("error: NLLB download produced no model.bin", file=sys.stderr)
            return 1
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
        print(f"nllb     ready, {size:.0f} MB  ({NLLB_REPO}, third party)")
        if not args.pair:
            return 0

    if not args.pair:
        ap.error("give at least one --pair, --nllb, or --list")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for pair in args.pair:
        if args.convert:
            failures += fetch_converted(pair, args.quantization)
        else:
            failures += fetch_verified(pair, args.force)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
