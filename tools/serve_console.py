#!/usr/bin/env python3
"""
Run the operator console.

Binds to localhost only. Caddy fronts it with basic auth, so a bind to 0.0.0.0
would publish an unauthenticated surface that can restart sessions and rewrite
settings straight onto the internet.

    LAD_TRANSLATE_PUBLIC_BASE=https://lad-translate-dev-...run.app \
    python tools/serve_console.py

The hardware output panel needs the database the session writes to:

    LAD_DATABASE_URL=postgresql://... LAD_CONTROL_SCHEMA=lad_translate_dev \
    LAD_TRANSLATE_TENANT=techiemaya python tools/serve_console.py

Without those three the rest of the console works and that panel says it is
not configured - it does not pretend the venue owns no devices.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.console.app import create_app
from lad_translate.obs.log import configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="localhost by default; Caddy is what faces outward")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--public-base", default=os.getenv("LAD_TRANSLATE_PUBLIC_BASE"),
                    help="where PHONES reach the join service - the Cloud Run URL")
    ap.add_argument("--env-file", type=Path, default=None)
    args = ap.parse_args()

    if not args.public_base:
        ap.error(
            "--public-base or LAD_TRANSLATE_PUBLIC_BASE is required: the QR "
            "codes must point at the join service a phone can reach, not at "
            "this box"
        )

    configure()
    import uvicorn

    app = create_app(public_base=args.public_base, env_path=args.env_file)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
