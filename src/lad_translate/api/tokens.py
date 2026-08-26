"""
Listener token service.

A listener scans a QR code, picks a language, and gets a LiveKit token for that
room. No app, no account, no login.

Three grant settings do the work, and `hidden` is the one that is easy to miss:

    can_publish=False       a listener has no microphone in this room. Without
                            it, 500 phones can publish audio into the room.

    can_publish_data=False  no data channel either. Nothing about this product
                            needs listeners to send anything.

    hidden=True             the listener is absent from the participant list.

That last one is a scale decision rather than a privacy one. A visible
participant joining is broadcast to everyone already in the room, so 500
visible listeners means 500 join events fanned out 500 ways. Hidden listeners
generate no such traffic. Risk area 4 is about whether a 500 participant
subscribe-only room holds up, and this setting is most of the answer.

Language tracks are NOT isolated from each other by the token. A listener with
a valid token can subscribe to any language track in the room. That is fine
here: every track carries a translation of the same public talk, so there is no
confidentiality boundary between them. It is written down so nobody later
assumes there is one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import SessionConfig
from ..obs.log import get_logger

log = get_logger(__name__)

DEFAULT_TTL_S = 6 * 60 * 60
"""
Token lifetime.

Long enough to outlast an event plus overruns, because a token expiring
mid-keynote drops that listener with no obvious way back other than rescanning.
"""


@dataclass(frozen=True, slots=True)
class ListenerToken:
    token: str
    room: str
    language: str
    listener_id: str
    url: str


class TokenIssuer:
    """Mints subscribe-only room tokens for listeners."""

    def __init__(
        self,
        livekit_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        ttl_s: int = DEFAULT_TTL_S,
        internal_url: str | None = None,
    ) -> None:
        self.livekit_url = livekit_url or os.getenv("LIVEKIT_URL")
        """
        What CLIENTS are told to dial. Behind a TLS proxy this is the public
        wss:// address, because a browser will not open an insecure WebSocket
        from a secure page.
        """

        self.internal_url = (
            internal_url or os.getenv("LIVEKIT_INTERNAL_URL") or self.livekit_url
        )
        """
        What SERVER-SIDE components dial: the translation service and any
        tooling running on the same host.

        These must not go through the proxy. The Python SDK does not trust
        Caddy's internal CA and fails with "invalid peer certificate:
        UnknownIssuer", and routing local traffic out through TLS to come
        straight back would be pointless even if it worked.
        """
        self._api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        self._api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")
        self.ttl_s = ttl_s

        missing = [
            name
            for name, value in (
                ("LIVEKIT_URL", self.livekit_url),
                ("LIVEKIT_API_KEY", self._api_key),
                ("LIVEKIT_API_SECRET", self._api_secret),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"missing LiveKit configuration: {', '.join(missing)}")

    # -------------------------------------------------------------------------

    def for_listener(self, room: str, language: str, listener_id: str) -> ListenerToken:
        """Subscribe-only, hidden, single room."""
        from livekit import api

        grants = api.VideoGrants(
            room_join=True,
            room=room,
            can_subscribe=True,
            can_publish=False,
            can_publish_data=False,
            hidden=True,
        )
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(f"listener-{listener_id}")
            .with_name(f"listener ({language})")
            # Carried so the join page knows which track to attach without a
            # second round trip.
            .with_attributes({"language": language})
            .with_grants(grants)
            .with_ttl(_timedelta(self.ttl_s))
            .to_jwt()
        )
        return ListenerToken(
            token=token,
            room=room,
            language=language,
            listener_id=listener_id,
            url=self.livekit_url,
        )

    def for_translator(self, config: SessionConfig) -> str:
        """
        Token for the translation service itself.

        Publishes the language tracks and subscribes to the speaker. It does
        not need room_admin: creating and destroying rooms is a control plane
        job, not something a media process should be able to do.
        """
        from livekit import api

        grants = api.VideoGrants(
            room_join=True,
            room=config.room_name,
            can_subscribe=True,
            can_publish=True,
            can_publish_data=False,
        )
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(f"translator-{config.session_id}")
            .with_name(f"LAD Translate ({config.event_name})")
            .with_grants(grants)
            .with_ttl(_timedelta(self.ttl_s))
            .to_jwt()
        )

    def for_publisher(self, room: str, identity: str = "venue-publisher") -> str:
        """
        Token for the venue laptop sending the speaker's microphone.

        Publishes only. It has no reason to subscribe, and not subscribing
        means the venue machine never pulls five translated streams back down
        a connection that is carrying the one that matters.
        """
        from livekit import api

        grants = api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=False,
            can_publish_data=False,
        )
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(identity)
            .with_name("Venue publisher")
            .with_grants(grants)
            .with_ttl(_timedelta(self.ttl_s))
            .to_jwt()
        )


def _timedelta(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=seconds)
