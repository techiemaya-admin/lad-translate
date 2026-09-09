"""
Google sign-in for the console.

Replaces basic auth, which cost a working afternoon: a 32 byte base64 password
with no trailing newline, copied out of a terminal where the shell prompt runs
onto the end of it, entered into a dialog whose username field is easy to leave
blank. Every one of those produced the same 401, and none of them said which.

Not IAP. IAP for a VM needs an HTTPS load balancer in front of it, and this box
also serves WebRTC media over raw UDP, which such a balancer does not carry. It
would mean a second hostname and address for the console alone.

FAILS CLOSED. An empty allowlist admits nobody rather than everybody, and a
missing client id disables the console rather than opening it. The previous
iteration of this file's neighbour - the Caddy config - took the SFU offline by
treating a missing optional secret as a reason to emit a broken directive, and
the lesson generalises: when configuration is absent, do less, not more.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass

from ..obs.log import get_logger

log = get_logger(__name__)

AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"

COOKIE = "lad_console_session"
SESSION_TTL_S = 12 * 60 * 60
"""
A working day. Long enough that an operator running an event is not signed out
mid-talk, short enough that a shared laptop does not stay signed in for ever.
"""


class AuthError(Exception):
    """Anything that should end as 'sign in again' rather than a stack trace."""


@dataclass(frozen=True)
class Config:
    client_id: str
    client_secret: str
    redirect_uri: str
    session_secret: str
    allowed_emails: frozenset[str]
    allowed_domains: frozenset[str]

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret and self.session_secret)

    def permits(self, email: str) -> bool:
        email = email.strip().lower()
        if email in self.allowed_emails:
            return True
        # The "@" has to be there. rpartition returns the WHOLE string as the
        # tail when the separator is absent, so a bare "techiemaya.com" matched
        # the domain allowlist and was admitted as if it were an address.
        local, at, domain = email.rpartition("@")
        return bool(at and local and domain) and domain in self.allowed_domains


def config_from_env() -> Config:
    def csv(name: str) -> frozenset[str]:
        raw = os.getenv(name, "")
        return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())

    return Config(
        client_id=os.getenv("CONSOLE_OAUTH_CLIENT_ID", ""),
        client_secret=os.getenv("CONSOLE_OAUTH_CLIENT_SECRET", ""),
        redirect_uri=os.getenv("CONSOLE_OAUTH_REDIRECT_URI", ""),
        session_secret=os.getenv("CONSOLE_SESSION_SECRET", ""),
        allowed_emails=csv("CONSOLE_ALLOWED_EMAILS"),
        allowed_domains=csv("CONSOLE_ALLOWED_DOMAINS"),
    )


# --- signed session cookie ---------------------------------------------------
#
# HMAC over a compact JSON payload. A cookie is all the state there is: no
# server-side session store to keep, and a restart does not sign everyone out.


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    """Never raises. Cookie contents are attacker-controlled, and a decode
    error here would be a 500 on every page rather than a refused session."""
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError):
        return b""


def issue_session(email: str, secret: str, now: float | None = None) -> str:
    payload = {"email": email, "exp": int((now or time.time()) + SESSION_TTL_S)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(sig)}"


def read_session(cookie: str | None, secret: str, now: float | None = None) -> str | None:
    """The signed-in email, or None. Never raises on malformed input."""
    if not cookie or not secret or "." not in cookie:
        return None
    body, _, sig = cookie.rpartition(".")
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    # compare_digest, not ==: a timing side channel here leaks the signature a
    # byte at a time, and this cookie is the only thing standing in front of a
    # console that can restart sessions.
    if not hmac.compare_digest(_unb64(sig), expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except ValueError:
        return None
    if float(payload.get("exp", 0)) < (now or time.time()):
        return None
    email = payload.get("email")
    return email if isinstance(email, str) else None


# --- the OAuth dance ---------------------------------------------------------


def authorize_url(cfg: Config, state: str, nonce: str) -> str:
    params = {
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "scope": "openid email",
        "state": state,
        "nonce": nonce,
        # The console is for a named operator, so ask every time rather than
        # silently reusing whichever account the browser happens to hold.
        "prompt": "select_account",
    }
    return f"{AUTHORIZE}?{urllib.parse.urlencode(params)}"


def new_state() -> str:
    return secrets.token_urlsafe(24)


async def exchange_and_verify(cfg: Config, code: str, nonce: str) -> str:
    """Swap the code for an ID token, verify it, and return the email."""
    import httpx
    from google.auth.transport import requests as ga_requests
    from google.oauth2 import id_token as ga_id_token

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            TOKEN,
            data={
                "code": code,
                "client_id": cfg.client_id,
                "client_secret": cfg.client_secret,
                "redirect_uri": cfg.redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if response.status_code != 200:
        raise AuthError("Google rejected the sign-in. Try again.")

    raw = response.json().get("id_token")
    if not raw:
        raise AuthError("Google returned no identity token.")

    try:
        # Verifies signature, issuer, audience and expiry against Google's keys.
        claims = ga_id_token.verify_oauth2_token(
            raw, ga_requests.Request(), cfg.client_id
        )
    except ValueError as exc:
        raise AuthError("Could not verify the identity token.") from exc

    if claims.get("nonce") != nonce:
        # Without this a token minted for another sign-in of the same client
        # could be replayed into this session.
        raise AuthError("Sign-in did not match this browser. Try again.")
    if not claims.get("email_verified"):
        raise AuthError("That Google account has no verified email address.")

    email = str(claims.get("email", "")).lower()
    if not cfg.permits(email):
        log.warning("console sign-in refused", extra={"email": email})
        raise AuthError(f"{email} is not permitted to use this console.")

    log.info("console sign-in", extra={"email": email})
    return email
