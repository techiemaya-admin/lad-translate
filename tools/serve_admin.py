#!/usr/bin/env python3
"""
Run the operator API for hardware audio output.

    ./tools/pg.sh start
    export LAD_DATABASE_URL=postgresql://lad@127.0.0.1:55432/salesmaya_agent
    export LAD_CONTROL_SCHEMA=lad_dev
    export LAD_ADMIN_TOKEN="$(openssl rand -hex 32)"
    python tools/serve_admin.py

Separate from tools/serve_join.py on purpose, and bound to localhost by
default. The join service is reachable by every phone in the room; this one
decides which language reaches which wire, and the two do not belong on the
same origin. Put it behind the venue's own network controls, or tunnel to it,
rather than binding it wide because the portal is on another host.

LAD_ADMIN_TOKEN has no default. Unset, every request is refused rather than
allowed: this is the credential LAD-Frontend presents, so a plausible default
would be a working password for anyone who read the source.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.api.admin import create_admin_app
from lad_translate.db.pool import control_schema, database_url
from lad_translate.obs.log import configure


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    configure()
    import uvicorn

    if not os.getenv("LAD_ADMIN_TOKEN"):
        print(
            "LAD_ADMIN_TOKEN is not set: every request will be refused.\n"
            "  export LAD_ADMIN_TOKEN=\"$(openssl rand -hex 32)\"",
            file=sys.stderr,
        )

    # The pool is opened inside the app lifespan, on uvicorn's loop. Building it
    # here would bind it to a loop that is closed before the first request.
    app = create_admin_app(
        control_schema=control_schema(),
        database_url=database_url(),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
