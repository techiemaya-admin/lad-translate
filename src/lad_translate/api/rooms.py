"""
Live room inspection.

The join page has two sources of truth about which languages exist, and they
can disagree:

    the session row      what the event was configured to offer
    the LiveKit room     what is actually being published right now

A listener who picks a language from the first that is missing from the second
gets a valid token, a successful connection, and silence. Nothing errors,
because nothing is wrong from the transport's point of view: they subscribed to
a track that does not exist.

That happened in testing. The session record listed French, Arabic and German;
the running session published only French and Arabic; the page offered all
three. German listeners sat on "waiting for audio" indefinitely.

This module asks the room what is really there.
"""

from __future__ import annotations

from ..obs.log import get_logger

log = get_logger(__name__)

TRACK_PREFIX = "lang-"

SOURCE_TRACK_NAME = "source-audio"
"""
The speaker's own track, published by the venue laptop.

Offered to listeners as the original language. It is a RELAY, not a
translation: no STT, no translation, no synthesis, so it carries none of the
transcription errors and none of the added latency. In a hall with poor
acoustics, or for someone hard of hearing, the floor audio in earbuds is worth
more than anything downstream of it.
"""


def http_url(livekit_url: str) -> str:
    """LiveKit's server API speaks HTTP, while clients are given a ws URL."""
    return livekit_url.replace("wss://", "https://").replace("ws://", "http://")


class RoomInspector:
    """Reads which language tracks a room is currently publishing."""

    def __init__(self, livekit_url: str, api_key: str, api_secret: str) -> None:
        self._url = http_url(livekit_url)
        self._key = api_key
        self._secret = api_secret

    async def published_languages(self, room: str) -> set[str]:
        """
        Language codes with a live track in this room.

        Returns an empty set rather than raising if the room cannot be reached.
        A failure to inspect must not stop people joining: the room may be fine
        and the inspection call the only thing broken, and refusing to serve a
        join page because a status query failed would turn a small fault into
        an event-wide one.
        """
        from livekit import api

        client = api.LiveKitAPI(self._url, self._key, self._secret)
        try:
            result = await client.room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
            published: set[str] = set()
            for participant in result.participants:
                for track in participant.tracks:
                    if track.name.startswith(TRACK_PREFIX):
                        published.add(track.name[len(TRACK_PREFIX):])
                    elif track.name == SOURCE_TRACK_NAME:
                        published.add(SOURCE_TRACK_NAME)
            return published
        except Exception:
            log.warning("could not inspect room, assuming unknown", extra={"room": room})
            return set()
        finally:
            await client.aclose()
