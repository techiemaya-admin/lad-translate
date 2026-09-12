"""
Session recording: the speaker and every translation, as aligned WAV files.

    recordings/<room>/<session_id>/
        manifest.json     who, when, which files, at what rates
        source.wav        the speaker, at the rate the room delivered
        fr.wav  ar.wav …  one per language, at the voice's rate

ALIGNED, NOT CONCATENATED. A language track is bursty - a phrase, then
nothing while the speaker keeps talking - and a recorder that appends phrase
after phrase produces a file that is shorter than the talk and lines up with
nothing. Every track here is written on the session's wall clock: a phrase
that arrives eight seconds after the last one is preceded by eight seconds
of silence. All the files end up the same length, and dropping them into any
editor puts French under English where it belongs. A DAW is the tool for
"what did the audience hear when"; this makes the files fit one.

The alignment is by wall time at the recorder, which is when a phrase was
handed to the playout queue, not when the audience heard it. The two differ
by the queue depth - a second or so, and up to a few seconds when the drift
controller is working - so a translation in the file sits slightly earlier
than it did in the room. Fine for the purpose; noted so nobody measures
latency off these files.

THE SPEAKER IS TAPPED BEFORE THE BACKLOG GUARD. The pipeline sheds source
audio when STT falls behind, and that is the right call for translation. It
is the wrong call for a recording, which is the one artefact that should hold
what was actually said. So the tap sits upstream of the guard: what the
guard drops is still in source.wav, and the file is the record of the talk,
not of the pipeline's good day.

IT SURVIVES A CRASH. Python's wave module writes the RIFF sizes at close, so
a process that dies mid-talk leaves a header claiming zero bytes and a file
nothing will open. The writer here patches the sizes into the header every
few seconds and at every stop, so the worst case is the last few seconds.

RECORDING IS A SINK, and one that is always installed but not always armed.
FanOutSink(room, recorder) gives the recorder every phrase the room gets,
through the same interface the AES67 sink uses, with the same rule that its
failure cannot end the session. Arming and disarming mid-session is then a
flag flip and a set of files, not a rebuild of the fan-out - which is what
lets a console button start a recording without restarting the talk.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..adapters.base import AudioFrame
from ..obs.log import get_logger

log = get_logger(__name__)

GAP_TOLERANCE_S = 0.25
"""Below this, a gap between what the clock says and what has been written is
jitter and the audio is appended contiguously. Above it, it is a real pause
and silence is written to keep the track on the clock."""

HEADER_PATCH_EVERY_S = 5.0
SOURCE_NAME = "source"
MANIFEST = "manifest.json"


class WavWriter:
    """
    A 16-bit mono WAV, written incrementally with a header that stays valid.

    The RIFF and data chunk sizes are patched in place on a timer and at
    close. Between patches the header is at most HEADER_PATCH_EVERY_S behind
    the data, which is what a crash costs.
    """

    HEADER_BYTES = 44

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.samples = 0
        self._file = open(path, "wb")  # noqa: SIM115 - held open for the session
        self._file.write(self._header(0))
        self._last_patch = time.monotonic()

    def _header(self, data_bytes: int) -> bytes:
        return b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE" + b"fmt " + struct.pack(
            "<IHHIIHH", 16, 1, 1, self.sample_rate, self.sample_rate * 2, 2, 16
        ) + b"data" + struct.pack("<I", data_bytes)

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._file.write(pcm)
        self.samples += len(pcm) // 2
        if time.monotonic() - self._last_patch >= HEADER_PATCH_EVERY_S:
            self.patch_header()

    def write_silence(self, samples: int) -> None:
        if samples <= 0:
            return
        # In 64 KB pieces: a reconnect gap of minutes must not become one
        # allocation of hundreds of megabytes.
        remaining = samples * 2
        chunk = b"\x00" * min(remaining, 65536)
        while remaining > 0:
            piece = chunk if remaining >= len(chunk) else chunk[:remaining]
            self._file.write(piece)
            remaining -= len(piece)
        self.samples += samples

    def patch_header(self) -> None:
        data_bytes = self.samples * 2
        pos = self._file.tell()
        self._file.seek(0)
        self._file.write(self._header(data_bytes))
        self._file.seek(pos)
        self._file.flush()
        self._last_patch = time.monotonic()

    @property
    def seconds(self) -> float:
        return self.samples / self.sample_rate if self.sample_rate else 0.0

    def close(self) -> None:
        if self._file.closed:
            return
        self.patch_header()
        self._file.close()


@dataclass
class _Track:
    name: str
    writer: WavWriter | None = None
    """Opened on the first audio, because the rate is only known then."""

    rate_hint: int | None = None


@dataclass
class Recording:
    """One recording of one session: a directory of aligned tracks."""

    directory: Path
    session_id: str
    room: str
    event_name: str
    languages: list[str]
    started_at: float = field(default_factory=time.monotonic)
    started_iso: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    tracks: dict[str, _Track] = field(default_factory=dict)
    bytes_written: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tracks[SOURCE_NAME] = _Track(SOURCE_NAME)
        for code in self.languages:
            self.tracks[code] = _Track(code)
        self.write_manifest()

    # -- writing ------------------------------------------------------------

    def write(self, track_name: str, pcm: bytes, sample_rate: int, at_wall: float | None = None) -> None:
        """Place `pcm` on the track at the session clock, padding a real gap."""
        if self.closed or not pcm:
            return
        # Tracks are created on first audio, whatever the name. A recording
        # armed before publish_languages() ran - the --record flag does exactly
        # that - would otherwise have no language tracks and silently keep
        # only the speaker.
        track = self.tracks.get(track_name)
        if track is None:
            track = self.tracks[track_name] = _Track(track_name)
            if track_name != SOURCE_NAME and track_name not in self.languages:
                self.languages.append(track_name)
        if track.writer is None:
            track.writer = WavWriter(self.directory / f"{track_name}.wav", sample_rate)
            track.rate_hint = sample_rate
        elif sample_rate != track.writer.sample_rate:
            # One rate per file. Piper's medium voices are all 22050 and a
            # room negotiates one rate for the source, so this is a bug
            # somewhere upstream rather than a case to handle gracefully.
            log.warning(
                "recording track rate changed; keeping the file's rate",
                extra={"track": track_name, "file_rate": track.writer.sample_rate, "got": sample_rate},
            )

        writer = track.writer
        now = at_wall if at_wall is not None else time.monotonic()
        target = int((now - self.started_at) * writer.sample_rate)
        gap = target - writer.samples
        if gap > GAP_TOLERANCE_S * writer.sample_rate:
            writer.write_silence(gap)
            self.bytes_written += gap * 2
        writer.write(pcm)
        self.bytes_written += len(pcm)

    # -- lifecycle ------------------------------------------------------------

    def write_manifest(self, final: bool = False) -> None:
        files = {}
        for name, track in self.tracks.items():
            if track.writer is None:
                continue
            files[f"{name}.wav"] = {
                "track": name,
                "sample_rate": track.writer.sample_rate,
                "seconds": round(track.writer.seconds, 2),
            }
        manifest = {
            "session_id": self.session_id,
            "room": self.room,
            "event_name": self.event_name,
            "languages": self.languages,
            "started_at": self.started_iso,
            "ended_at": datetime.now(UTC).isoformat(timespec="seconds") if final else None,
            "seconds": round(time.monotonic() - self.started_at, 1),
            "bytes": self.bytes_written,
            "files": files,
            "aligned": True,
            "note": (
                "Every track is on the session's wall clock with silence for gaps, so the "
                "files line up sample for sample in any editor. A translation sits where it "
                "was handed to playout, slightly before the audience heard it."
            ),
        }
        (self.directory / MANIFEST).write_text(json.dumps(manifest, indent=2))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for track in self.tracks.values():
            if track.writer is not None:
                track.writer.close()
        self.write_manifest(final=True)

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started_at

    def summary(self) -> dict:
        return {
            "directory": str(self.directory),
            "seconds": round(self.seconds, 1),
            "bytes": self.bytes_written,
            "tracks": [name for name, t in self.tracks.items() if t.writer is not None],
        }


class RecordingSink:
    """
    An AudioSink that records when armed and does nothing when not.

    Always installed as a secondary of FanOutSink, so a recording can start
    and stop mid-session without touching the fan-out. Its queue_depth is
    zero on purpose: a file has no playout queue, and reporting anything else
    would feed the drift controller a number that means nothing.
    """

    def __init__(self, root: Path, session_id: str, room: str, event_name: str) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.room = room
        self.event_name = event_name
        self.languages: list[str] = []
        self.recording: Recording | None = None
        self.takes = 0
        """Recordings started this session. Each gets its own directory, so
        stopping and starting again does not overwrite the first take."""

    # -- AudioSink --------------------------------------------------------------

    async def publish_languages(self, languages: list[str]) -> None:
        self.languages = list(languages)
        if self.recording is not None:
            for code in languages:
                self.recording.tracks.setdefault(code, _Track(code))
            self.recording.languages = list(languages)
            self.recording.write_manifest()

    async def push(self, language: str, pcm: bytes, sample_rate: int) -> None:
        if self.recording is not None:
            self.recording.write(language, pcm, sample_rate)

    def queue_depth(self, language: str) -> float:
        return 0.0

    async def close(self) -> None:
        self.stop()

    # -- the tap ------------------------------------------------------------------

    async def tap(self, frames):
        """Pass source frames through, recording each one when armed."""
        async for frame in frames:
            if self.recording is not None:
                self._source(frame)
            yield frame

    def _source(self, frame: AudioFrame) -> None:
        self.recording.write(SOURCE_NAME, frame.pcm, frame.sample_rate, at_wall=frame.t_wall)

    # -- arming --------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.recording is not None

    def start(self) -> Recording:
        """Begin a take. Idempotent: a second start returns the running one."""
        if self.recording is not None:
            return self.recording
        self.takes += 1
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = self.root / self.room / f"{self.session_id}-{stamp}"
        self.recording = Recording(
            directory=directory,
            session_id=self.session_id,
            room=self.room,
            event_name=self.event_name,
            languages=self.languages,
        )
        log.info(
            "recording started",
            extra={"directory": str(directory), "languages": self.languages, "take": self.takes},
        )
        return self.recording

    def stop(self) -> dict | None:
        """End the take. Idempotent. Returns its summary, or None if none ran."""
        if self.recording is None:
            return None
        recording, self.recording = self.recording, None
        recording.close()
        summary = recording.summary()
        log.info("recording stopped", extra=summary)
        return summary

    def summary(self) -> dict:
        return {
            "recording": self.recording is not None,
            "directory": str(self.recording.directory) if self.recording else None,
            "takes": self.takes,
        }
