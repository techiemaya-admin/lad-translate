"""
Listener token tests.

The grants are the only thing standing between a public QR code and 500 phones
with a microphone in the room, so they are asserted rather than assumed.
"""

from __future__ import annotations

import time

import jwt
import pytest

from lad_translate.api.tokens import TokenIssuer
from tests.fakes import session_config

KEY, SECRET = "devkey", "lad-test-signing-key-not-a-real-secret"


@pytest.fixture
def issuer():
    return TokenIssuer(livekit_url="ws://127.0.0.1:7880", api_key=KEY, api_secret=SECRET)


def claims(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=["HS256"], options={"verify_aud": False})


def test_missing_configuration_is_refused_not_defaulted():
    with pytest.raises(RuntimeError, match="missing LiveKit configuration"):
        TokenIssuer(livekit_url=None, api_key=None, api_secret=None)


# --- listener ---------------------------------------------------------------


def test_listener_cannot_publish_audio(issuer):
    """Otherwise a public QR code hands 500 phones a microphone in the room."""
    video = claims(issuer.for_listener("room-1", "fr", "abc").token)["video"]
    assert video["canPublish"] is False
    assert video["canPublishData"] is False
    assert video["canSubscribe"] is True


def test_listener_is_hidden_from_the_participant_list(issuer):
    """
    A scale decision, not a privacy one.

    Every visible participant joining is broadcast to everyone already in the
    room, so 500 visible listeners means 500 joins fanned out 500 ways. Risk
    area 4 is whether a 500 participant subscribe-only room holds up, and this
    is most of the answer.
    """
    assert claims(issuer.for_listener("room-1", "fr", "abc").token)["video"]["hidden"] is True


def test_listener_is_confined_to_one_room(issuer):
    video = claims(issuer.for_listener("room-1", "fr", "abc").token)["video"]
    assert video["room"] == "room-1"
    assert video["roomJoin"] is True
    assert not video.get("roomCreate")
    assert not video.get("roomAdmin")


def test_listener_language_travels_on_the_token(issuer):
    """Saves the join page a round trip to find out which track to attach."""
    decoded = claims(issuer.for_listener("room-1", "ar", "abc").token)
    assert decoded["attributes"]["language"] == "ar"


def test_listener_identity_is_unique_per_listener(issuer):
    a = claims(issuer.for_listener("room-1", "fr", "one").token)["sub"]
    b = claims(issuer.for_listener("room-1", "fr", "two").token)["sub"]
    assert a != b


def test_token_outlasts_a_long_event(issuer):
    """A token expiring mid-keynote drops that listener with no way back."""
    decoded = claims(issuer.for_listener("room-1", "fr", "abc").token)
    # LiveKit issues nbf/exp, not iat, so measure the remaining life.
    assert decoded["exp"] - decoded["nbf"] >= 4 * 60 * 60
    assert decoded["exp"] - time.time() >= 4 * 60 * 60


# --- translator and publisher ----------------------------------------------


def test_translator_publishes_and_subscribes_but_is_not_an_admin(issuer):
    """Creating and destroying rooms is a control plane job, not a media one."""
    video = claims(issuer.for_translator(session_config()))["video"]
    assert video["canPublish"] is True
    assert video["canSubscribe"] is True
    assert not video.get("roomAdmin")
    assert not video.get("roomCreate")


def test_venue_publisher_does_not_subscribe(issuer):
    """
    The venue laptop has no reason to pull five translated streams back down
    the connection that is carrying the one that matters.
    """
    video = claims(issuer.for_publisher("room-1"))["video"]
    assert video["canPublish"] is True
    assert video["canSubscribe"] is False


def test_every_role_is_scoped_to_its_room(issuer):
    config = session_config()
    for token in (
        issuer.for_listener(config.room_name, "fr", "abc").token,
        issuer.for_translator(config),
        issuer.for_publisher(config.room_name),
    ):
        assert claims(token)["video"]["room"] == config.room_name
