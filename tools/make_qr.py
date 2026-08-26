#!/usr/bin/env python3
"""
Print the QR code for a session's join page.

The QR must point at a host the audience's phones can actually reach. On a
venue network that is not localhost, and it is worth testing on a real phone
on the venue wifi rather than assuming: guest networks often use client
isolation, which lets a phone reach the internet but not another device on the
same LAN.

Usage:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--base", help="default: http://<lan ip>:8080")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--png", type=Path, help="write a PNG here (default: fixtures/qr.png)")
    ap.add_argument("--ascii", action="store_true", help="also print the terminal QR")
    args = ap.parse_args()

    base = args.base or f"http://{local_ip()}:{args.port}"
    url = f"{base.rstrip('/')}/s/{args.session}"

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

    png = args.png or (Path(__file__).resolve().parent.parent / "fixtures" / "qr.png")
    png.parent.mkdir(parents=True, exist_ok=True)
    qr.make_image(fill_color="black", back_color="white").save(str(png))
    print(f"\n{url}")
    print(f"QR image: {png}\n")

    if args.ascii:
        qr.print_ascii(invert=True)
    if "127.0.0.1" in url or "localhost" in url:
        print("WARNING  This URL only works on this machine. Pass --base with a")
        print("         LAN address the audience's phones can reach.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
