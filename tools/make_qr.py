#!/usr/bin/env python3
"""
Print the QR code for a session's join or speak page.

The QR must point at a host the audience's phones can actually reach. On a
venue network that is not localhost, and it is worth testing on a real phone
on the venue wifi rather than assuming: guest networks often use client
isolation, which lets a phone reach the internet but not another device on the
same LAN.

Prefer --room over --session. A code printed against a session id dies the
moment the service restarts, and codes go on badges and signage days before an
event; a room URL resolves to whatever session is live in that room.

Usage:
    python tools/make_qr.py --room demo-room --base https://192.168.1.20:8443
    python tools/make_qr.py --room demo-room --speak --base https://192.168.1.20:8443
    python tools/make_qr.py --session <uuid> --base http://192.168.1.20:8080
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path


def local_ip() -> str:
    """Best guess at this machine's LAN address. No packets are sent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 1))  # TEST-NET-1, never routed
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def target_path(args) -> str:
    """The path for this combination of room/session and listen/speak.

    Four routes, and the room forms are the ones that survive a restart:

        /room/<name>          /room/<name>/speak
        /s/<session-id>       /speak/<session-id>
    """
    if args.room:
        return f"/room/{args.room}/speak" if args.speak else f"/room/{args.room}"
    return f"/speak/{args.session}" if args.speak else f"/s/{args.session}"


def default_png(args) -> Path:
    who = args.room or args.session
    kind = "speak" if args.speak else "listen"
    return Path(__file__).resolve().parent.parent / "fixtures" / f"qr-{kind}-{who}.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--room", help="stable room name; survives a restart")
    target.add_argument("--session", help="one specific session id")
    ap.add_argument("--speak", action="store_true", help="the speaker page, not the listener page")
    ap.add_argument("--base", help="default: http://<lan ip>:8080")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--png", type=Path, help="write a PNG here")
    ap.add_argument("--ascii", action="store_true", help="also print the terminal QR")
    args = ap.parse_args()

    base = args.base or f"http://{local_ip()}:{args.port}"
    url = f"{base.rstrip('/')}{target_path(args)}"

    import qrcode

    # A PNG, not terminal art. ASCII QR codes depend on the font having square
    # cells and on the terminal's colours matching what the encoder assumed;
    # phones frequently cannot read them. An image always scans.
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    png = args.png or default_png(args)
    png.parent.mkdir(parents=True, exist_ok=True)
    qr.make_image(fill_color="black", back_color="white").save(str(png))
    print(f"\n{url}")
    print(f"QR image: {png}\n")

    if args.ascii:
        qr.print_ascii(invert=True)
    if "127.0.0.1" in url or "localhost" in url:
        print("WARNING  This URL only works on this machine. Pass --base with a")
        print("         LAN address the audience's phones can reach.\n")
    # The speaker page needs a microphone, and browsers only expose one in a
    # secure context. On plain http from anything but localhost
    # navigator.mediaDevices is undefined, so the page loads and simply has no
    # microphone API to call -- which looks like a broken page rather than a
    # missing certificate. Run tools/tls.sh (or tls.ps1) and point --base at it.
    if args.speak and url.startswith("http://") and "127.0.0.1" not in url and "localhost" not in url:
        print("WARNING  A speaker page over plain http has no microphone. Browsers")
        print("         expose one only in a secure context. Start the TLS proxy and")
        print("         pass --base https://<lan ip>:8443\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
