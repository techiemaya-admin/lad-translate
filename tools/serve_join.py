#!/usr/bin/env python3
"""
Run the listener join service.

    ./tools/pg.sh start
    ./tools/livekit.sh start
    export LAD_DATABASE_URL=postgresql://lad@127.0.0.1:55432/salesmaya_agent
    export LAD_CONTROL_SCHEMA=lad_dev
    export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
    python tools/serve_join.py

Bind to 0.0.0.0 to reach it from a phone on the same network, which is the only
way to find out whether the page really works: desktop Chrome does not enforce
the autoplay rules that decide this on iOS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.api.join import create_app  # noqa: E402
from lad_translate.api.tokens import TokenIssuer  # noqa: E402
from lad_translate.db.pool import control_schema, database_url  # noqa: E402
from lad_translate.obs.log import configure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from a phone")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    configure()
    import uvicorn

    # The pool is opened inside the app lifespan, on uvicorn's loop. Building
    # it here with asyncio.run() would bind it to a loop that is closed before
    # the first request arrives.
    app = create_app(
        issuer=TokenIssuer(),
        control_schema=control_schema(),
        database_url=database_url(),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
