"""
LiveKit room transport.

Subscribes to the speaker's track and publishes one audio track per target
language into the same room. Listeners subscribe to exactly one of those.

Fan-out cost scales with language count, not audience size. Only WebRTC egress
scales per listener, and that is the SFU's problem rather than this process's.

On queue sizing: rtc.AudioSource paces itself, blocking capture_frame once its
buffer is full. The default buffer is one second, which is smaller than the
drift thresholds in session/drift.py, so the source would block before the
controller ever saw a queue worth acting on. It is opened wide here on purpose
and `queued_duration` is read as the drift signal instead. The queue is the
drift: audio synthesised but not yet spoken is exactly how far behind that
language is.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..adapters.base import AudioFrame
from ..obs.log import get_logger

log = get_logger(__name__)

SOURCE_TRACK_NAME = "source-audio"
"""Track the venue publisher sends the speaker's microphone on."""

SPEECH_BITRATE = 32_000
"""
Opus bitrate per language track, bits per second.

Left unset, LiveKit publishes a microphone source at a bitrate meant for
general audio. 32 kbps mono Opus is comfortable for speech, and these tracks
only ever carry one synthesised voice: no music, no room tone, no second
speaker.

MEASURED on a real browser listener, same setup each time:

    default (nothing set)      98.7 kbps
    32 kbps cap, RED on        66.9 kbps
    32 kbps cap, RED off       32.4 kbps

Per 500 listeners that is 49 Mbps, 33 Mbps and 16 Mbps of egress.
"""

SPEECH_RED = True
"""
Opus redundancy. Costs almost exactly 2x, see the table above.

On by default. Venue wifi under load drops packets, and RED lets the decoder
reconstruct a lost one from the next: without it a drop is an audible gap in
someone's translation. 67 kbps still sits near the roughly 50 kbps the
architecture budgets, and the brief's own constraint is access point client
capacity rather than bandwidth.

Turn it off for a venue with a known-good wired uplink and a lot of listeners,
where halving egress matters more than loss resilience. Do not turn it off
blind: the failure it prevents is intermittent and hard to reproduce after the
event.
"""

PLAYOUT_QUEUE_MS = 12_000
"""
Publish buffer per language.

Must exceed DriftPolicy.skip_at_s, or the source blocks before the drift
controller can act and the whole fan-out stalls behind one slow language.
"""


@dataclass(slots=True)
class LanguageTrack:
    language: str
    source: object
    track: object
    publication: object | None = None
    frames_published: int = 0
    seconds_published: float = 0.0


class TranslationRoom:
    """One room: one source track in, N language tracks out."""

    def __init__(self, room_name: str, sample_rate: int, num_channels: int = 1) -> None:
        self.room_name = room_name
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self._room = None
        self._tracks: dict[str, LanguageTrack] = {}
        self._source_track = None
        self._source_ready = asyncio.Event()
        self._closing = False
        """Set before a deliberate disconnect, so shutdown is not reported as a fault."""

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, url: str, token: str) -> None:
        import livekit.rtc as rtc

        self._room = rtc.Room()

        @self._room.on("track_published")
        def _on_published(publication, participant):  # noqa: ANN001
            # With auto_subscribe off, nothing arrives unless it is asked for.
            self._subscribe_if_source(publication, participant)

        @self._room.on("track_subscribed")
        def _on_subscribed(track, publication, participant):  # noqa: ANN001
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if publication.name != SOURCE_TRACK_NAME:
                # Should be unreachable with auto_subscribe off. If it happens,
                # drop it rather than decode audio nobody asked for.
                publication.set_subscribed(False)
                return
            self._source_track = track
            self._source_ready.set()
            log.info(
                "source track subscribed",
                extra={"track_name": publication.name, "participant": participant.identity},
            )

        @self._room.on("disconnected")
        def _on_disconnected(reason):  # noqa: ANN001
            self._source_ready.clear()
            if self._closing:
                log.info("room disconnected", extra={"reason": str(reason)})
                return
            # Losing the room mid-event takes the whole audience offline, so it
            # is worth an alert. A planned shutdown is not, and crying wolf on
            # every clean stop trains people to ignore the alert that matters.
            log.error("room disconnected unexpectedly", extra={"reason": str(reason)})

        # auto_subscribe defaults to True, which would subscribe this process
        # to every track in the room INCLUDING the N language tracks it
        # publishes itself. That is the translation service downloading its own
        # output back from the SFU: N times 32kbps of pure waste plus the decode
        # cost, on the machine that is already the bottleneck. Ignoring those
        # tracks after the fact treats the symptom; not subscribing is the fix.
        await self._room.connect(url, token, rtc.RoomOptions(auto_subscribe=False))

        # Anything already published before we joined never fires
        # track_published, so sweep once on connect.
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                self._subscribe_if_source(publication, participant)

        log.info("room connected", extra={"room": self.room_name})

    def _subscribe_if_source(self, publication, participant) -> None:  # noqa: ANN001
        """Subscribe to the speaker's track and nothing else."""
        if publication.name != SOURCE_TRACK_NAME:
            return
        if not publication.subscribed:
            publication.set_subscribed(True)
            log.info(
                "subscribing to source track",
                extra={"track_name": publication.name, "participant": participant.identity},
            )

    async def close(self) -> None:
        self._closing = True
        for entry in self._tracks.values():
            await entry.source.aclose()
        if self._room is not None:
            await self._room.disconnect()
        log.info(
            "room closed",
            extra={
                "room": self.room_name,
                "published": {k: round(v.seconds_published, 1) for k, v in self._tracks.items()},
            },
        )

    # -------------------------------------------------------------------------
    # Source audio
    # -------------------------------------------------------------------------

    async def wait_for_source(self, timeout_s: float = 60.0) -> None:
        """Block until the venue publisher appears, or give up and say so."""
        try:
            await asyncio.wait_for(self._source_ready.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"no {SOURCE_TRACK_NAME!r} track in room {self.room_name} after "
                f"{timeout_s}s; is the venue publisher connected?"
            ) from exc

    async def source_frames(self) -> AsyncIterator[AudioFrame]:
        """
        Stream the speaker's audio.

        t_audio accumulates from the frames themselves rather than from the
        clock, so it stays correct if this process is briefly descheduled.
        """
        import livekit.rtc as rtc

        await self.wait_for_source()
        stream = rtc.AudioStream.from_track(track=self._source_track)
        t_audio = 0.0
        try:
            async for event in stream:
                frame = event.frame
                yield AudioFrame(
                    pcm=bytes(frame.data),
                    sample_rate=frame.sample_rate,
                    t_audio=t_audio,
                    t_wall=time.monotonic(),
                )
                t_audio += frame.samples_per_channel / frame.sample_rate
        finally:
            await stream.aclose()

    # -------------------------------------------------------------------------
    # Language tracks
    # -------------------------------------------------------------------------

    async def publish_languages(self, languages: list[str]) -> None:
        import livekit.rtc as rtc

        # AudioEncoding is not re-exported on livekit.rtc, only on the
        # generated protobuf module. Pinned here so an SDK upgrade that moves
        # it fails loudly rather than silently reverting to the default
        # bitrate, which is the bug this whole block exists to prevent.
        from livekit.rtc._proto.room_pb2 import AudioEncoding

        for language in languages:
            source = rtc.AudioSource(
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
                queue_size_ms=PLAYOUT_QUEUE_MS,
            )
            # Track name is what the listener join page selects on.
            track = rtc.LocalAudioTrack.create_audio_track(f"lang-{language}", source)
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE,
                audio_encoding=AudioEncoding(max_bitrate=SPEECH_BITRATE),
                # DTX stops sending during silence, which is most of a
                # translated stream: each language is quiet while the others
                # speak. RED adds redundancy for packet loss, which venue wifi
                # produces plenty of, at a small bitrate cost worth paying.
                dtx=True,
                red=SPEECH_RED,
            )
            publication = await self._room.local_participant.publish_track(track, options)
            self._tracks[language] = LanguageTrack(
                language=language, source=source, track=track, publication=publication
            )
            log.info(
                "language track published",
                extra={"language": language, "track_name": f"lang-{language}"},
            )

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        """Hand synthesised audio to a language track."""
        import livekit.rtc as rtc

        entry = self._tracks.get(language)
        if entry is None:
            raise KeyError(f"no published track for {language!r}")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{language} audio is {sample_rate}Hz but the track is "
                f"{self.sample_rate}Hz; resample before publishing"
            )

        samples = len(pcm) // 2 // self.num_channels
        if samples == 0:
            return
        await entry.source.capture_frame(
            rtc.AudioFrame(
                data=pcm,
                sample_rate=sample_rate,
                num_channels=self.num_channels,
                samples_per_channel=samples,
            )
        )
        entry.frames_published += 1
        entry.seconds_published += samples / sample_rate

    def queue_depth(self, language: str) -> float:
        """
        Seconds of audio synthesised but not yet spoken.

        This is the drift signal. See session/drift.py.
        """
        entry = self._tracks.get(language)
        return float(entry.source.queued_duration) if entry else 0.0

    def clear_queue(self, language: str) -> None:
        """
        Discard everything pending for one language.

        Only for a chain so far behind that its audio is no longer related to
        what the speaker is saying. Loses speech, and is logged by the caller.
        """
        entry = self._tracks.get(language)
        if entry is not None:
            entry.source.clear_queue()

    @property
    def languages(self) -> list[str]:
        return list(self._tracks)
