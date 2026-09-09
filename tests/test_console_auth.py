"""
Google sign-in for the console.

The tests that matter are the refusals. This cookie is the only thing in front
of a console that can restart sessions and rewrite settings, and the thing it
replaced - basic auth - failed open in the sense that mattered: it produced one
indistinguishable 401 for a wrong username, a mistyped password, and a password
copied with a shell prompt attached.
"""

from __future__ import annotations

import time

import pytest

from lad_translate.console import auth

SECRET = "test-session-secret"


def cfg(**over) -> auth.Config:
    base = dict(
        client_id="cid.apps.googleusercontent.com",
        client_secret="csecret",
        redirect_uri="https://host/console/auth/callback",
        session_secret=SECRET,
        allowed_emails=frozenset({"someone@example.com"}),
        allowed_domains=frozenset({"techiemaya.com"}),
    )
    base.update(over)
    return auth.Config(**base)


# --- who is let in ----------------------------------------------------------

def test_an_allowed_domain_is_permitted():
    assert cfg().permits("naveen@techiemaya.com")
    assert cfg().permits("NAVEEN@TechieMaya.com"), "case must not decide access"


def test_an_explicitly_listed_address_is_permitted():
    assert cfg().permits("someone@example.com")


def test_everyone_else_is_refused():
    assert not cfg().permits("stranger@gmail.com")
    assert not cfg().permits("naveen@techiemaya.com.attacker.test")
    assert not cfg().permits("techiemaya.com")       # no @, not a domain match
    assert not cfg().permits("")


def test_an_empty_allowlist_admits_nobody():
    """
    Fails closed. The Caddy incident earlier today came from treating missing
    configuration as permission to carry on; an empty allowlist here must mean
    nobody rather than everybody.
    """
    empty = cfg(allowed_emails=frozenset(), allowed_domains=frozenset())
    assert not empty.permits("naveen@techiemaya.com")
    assert not empty.permits("anyone@anywhere.test")


def test_missing_credentials_disable_the_console():
    assert not cfg(client_id="").enabled
    assert not cfg(client_secret="").enabled
    assert not cfg(session_secret="").enabled
    assert cfg().enabled


# --- the session cookie -----------------------------------------------------

def test_a_session_round_trips():
    cookie = auth.issue_session("naveen@techiemaya.com", SECRET)
    assert auth.read_session(cookie, SECRET) == "naveen@techiemaya.com"


def test_a_tampered_payload_is_refused():
    """The whole point of signing it."""
    cookie = auth.issue_session("nobody@example.com", SECRET)
    body, _, sig = cookie.rpartition(".")
    forged = auth._b64(b'{"email":"admin@techiemaya.com","exp":9999999999}')
    assert auth.read_session(f"{forged}.{sig}", SECRET) is None


def test_a_signature_from_another_secret_is_refused():
    cookie = auth.issue_session("naveen@techiemaya.com", "someone-elses-secret")
    assert auth.read_session(cookie, SECRET) is None


def test_an_expired_session_is_refused():
    old = auth.issue_session("naveen@techiemaya.com", SECRET,
                             now=time.time() - auth.SESSION_TTL_S - 10)
    assert auth.read_session(old, SECRET) is None


@pytest.mark.parametrize("junk", [None, "", "not-a-cookie", "a.b", "....", "x." * 40])
def test_malformed_cookies_never_raise(junk):
    """
    A crash here is a 500 on every page. Whatever arrives in a cookie is
    attacker-controlled and must only ever produce None.
    """
    assert auth.read_session(junk, SECRET) is None


# --- the authorize URL ------------------------------------------------------

def test_the_authorize_url_carries_state_and_nonce():
    url = auth.authorize_url(cfg(), "STATE123", "NONCE456")
    assert "state=STATE123" in url and "nonce=NONCE456" in url
    assert "scope=openid+email" in url
    assert "prompt=select_account" in url, "a named operator should choose the account"


def test_state_values_are_unpredictable():
    values = {auth.new_state() for _ in range(200)}
    assert len(values) == 200
    assert all(len(v) >= 24 for v in values)
