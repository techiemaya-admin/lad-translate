"""
Session recording: aligned WAVs that survive a crash.

Judged by reading the files back with the standard library's wave module -
if that opens them, an editor will - and by where the audio lands in time,
which is the property that makes the files worth having.
"""

from __future__ import annotations

import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from lad_translate.adapters.base import AudioFrame
from lad_translate.session import recording as rec
from lad_translate.session.recording import Recording, RecordingSink, WavWriter
from lad_translate.session.sinks import AudioSink, FanOutSink


def tone(seconds: float, rate: int, hz: float = 440.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * hz * t) * 0.5 * 32767).astype(np.int16).tobytes()


def read(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


# --- the writer ---------------------------------------------------------------


def test_a_wav_reads_back_with_the_right_length_and_rate(tmp_path: Path):
    w = WavWriter(tmp_path / "a.wav", 22050)
    w.write(tone(0.5, 22050))
    w.close()
    rate, samples = read(tmp_path / "a.wav")
    assert rate == 22050
    assert samples.size == int(0.5 * 22050)


def test_the_header_is_valid_before_close(tmp_path: Path):
    """
    The crash case. wave.open on a file whose header says zero bytes raises
    or returns nothing; a patched header returns everything written so far.
    """
    w = WavWriter(tmp_path / "b.wav", 16000)
    w.write(tone(1.0, 16000))
    w.patch_header()
    # Not closed. Read a copy of the bytes as they are on disk right now.
    w._file.flush()
    rate, samples = read(tmp_path / "b.wav")
    assert samples.size == 16000
    w.close()


def test_the_header_patches_itself_on_a_timer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rec, "HEADER_PATCH_EVERY_S", 0.0)
    w = WavWriter(tmp_path / "c.wav", 8000)
    w.write(tone(0.25, 8000))
    w._file.flush()
    raw = (tmp_path / "c.wav").read_bytes()
    data_bytes = struct.unpack("<I", raw[40:44])[0]
    assert data_bytes == 0.25 * 8000 * 2
    w.close()


def test_silence_is_written_in_pieces_not_one_allocation(tmp_path: Path):
    w = WavWriter(tmp_path / "d.wav", 48000)
    w.write_silence(48000 * 30)  # thirty seconds
    w.close()
    rate, samples = read(tmp_path / "d.wav")
    assert samples.size == 48000 * 30
    assert not samples.any()


# --- alignment -------------------------------------------------------------------


def test_tracks_are_placed_on_the_session_clock(tmp_path: Path):
    """
    A phrase four seconds in lands four seconds in, with silence before it.
    That is the difference between a recording and a pile of phrases.
    """
    r = Recording(tmp_path / "s", "sid", "hall", "Test", ["fr"])
    t0 = r.started_at
    r.write("fr", tone(0.5, 22050), 22050, at_wall=t0 + 4.0)
    r.close()
    rate, samples = read(tmp_path / "s" / "fr.wav")
    assert rate == 22050
    lead = samples[: int(3.9 * 22050)]
    body = samples[int(4.0 * 22050) : int(4.5 * 22050)]
    assert not lead.any(), "the four seconds before the phrase should be silence"
    assert np.abs(body).max() > 10000, "the phrase should be where the clock put it"
    assert abs(samples.size - int(4.5 * 22050)) <= 2


def test_jitter_is_appended_but_a_pause_is_padded(tmp_path: Path):
    r = Recording(tmp_path / "s", "sid", "hall", "Test", ["fr"])
    t0 = r.started_at
    r.write("fr", tone(1.0, 22050), 22050, at_wall=t0)
    # 1.1s in: 0.1s late against 1.0s written - jitter, appended contiguously.
    r.write("fr", tone(0.2, 22050), 22050, at_wall=t0 + 1.1)
    assert r.tracks["fr"].writer.samples == int(1.2 * 22050)
    # 5.0s in: a real pause - padded to the clock.
    r.write("fr", tone(0.2, 22050), 22050, at_wall=t0 + 5.0)
    assert abs(r.tracks["fr"].writer.samples - int(5.2 * 22050)) <= 2
    r.close()


def test_audio_arriving_ahead_of_the_clock_is_never_truncated(tmp_path: Path):
    """A burst after a speed-up lands ahead of wall time; it is kept, not cut."""
    r = Recording(tmp_path / "s", "sid", "hall", "Test", ["ar"])
    t0 = r.started_at
    r.write("ar", tone(3.0, 22050), 22050, at_wall=t0 + 0.5)   # 0.5s pad + 3.0s = 3.5s
    r.write("ar", tone(1.0, 22050), 22050, at_wall=t0 + 1.0)   # clock says 1.0, file is at 3.5
    assert r.tracks["ar"].writer.samples == int(4.5 * 22050)
    r.close()


def test_source_and_languages_end_up_the_same_length(tmp_path: Path):
    r = Recording(tmp_path / "s", "sid", "hall", "Test", ["fr", "de"])
    t0 = r.started_at
    for i in range(10):  # ten seconds of source, in 100ms frames
        for j in range(10):
            r.write("source", tone(0.1, 48000), 48000, at_wall=t0 + i + j / 10)
    r.write("fr", tone(1.0, 22050), 22050, at_wall=t0 + 3.0)
    r.write("de", tone(1.0, 22050), 22050, at_wall=t0 + 9.0)
    r.close()
    src = read(tmp_path / "s" / "source.wav")[1].size / 48000
    fr = read(tmp_path / "s" / "fr.wav")[1].size / 22050
    de = read(tmp_path / "s" / "de.wav")[1].size / 22050
    assert abs(src - 10.0) < 0.15
    assert abs(fr - 4.0) < 0.05 and abs(de - 10.0) < 0.05


def test_the_manifest_describes_the_files(tmp_path: Path):
    r = Recording(tmp_path / "s", "sid-1", "hall-a", "Keynote", ["fr"])
    assert (tmp_path / "s" / "manifest.json").exists(), "written at start, so a crash leaves one"
    r.write("source", tone(0.5, 48000), 48000, at_wall=r.started_at)
    r.write("fr", tone(0.5, 22050), 22050, at_wall=r.started_at)
    r.close()
    m = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert m["session_id"] == "sid-1" and m["room"] == "hall-a" and m["event_name"] == "Keynote"
    assert m["ended_at"] is not None and m["aligned"] is True
    assert m["files"]["source.wav"]["sample_rate"] == 48000
    assert m["files"]["fr.wav"]["sample_rate"] == 22050
    assert m["bytes"] == 0.5 * 48000 * 2 + 0.5 * 22050 * 2


# --- the sink ------------------------------------------------------------------------


def test_it_is_an_audio_sink_with_no_playout_queue():
    sink = RecordingSink(Path("/tmp"), "sid", "hall", "Test")
    assert isinstance(sink, AudioSink)
    assert sink.queue_depth("fr") == 0.0


@pytest.mark.asyncio
async def test_disarmed_it_writes_nothing_and_armed_it_writes_everything(tmp_path: Path):
    sink = RecordingSink(tmp_path, "sid", "hall", "Test")
    await sink.publish_languages(["fr"])
    await sink.push("fr", tone(0.2, 22050), 22050)
    assert not list(tmp_path.rglob("*.wav")), "nothing should be written while disarmed"

    sink.start()
    await sink.push("fr", tone(0.2, 22050), 22050)
    summary = sink.stop()
    assert summary is not None and "fr" in summary["tracks"]
    files = list(tmp_path.rglob("fr.wav"))
    assert len(files) == 1
    assert read(files[0])[1].size == int(0.2 * 22050)


@pytest.mark.asyncio
async def test_each_take_gets_its_own_directory(tmp_path: Path):
    sink = RecordingSink(tmp_path, "sid", "hall", "Test")
    await sink.publish_languages(["fr"])
    first = sink.start().directory
    sink.stop()
    time.sleep(1.05)  # the directory stamp is to the second
    second = sink.start().directory
    sink.stop()
    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.asyncio
async def test_start_and_stop_are_idempotent(tmp_path: Path):
    sink = RecordingSink(tmp_path, "sid", "hall", "Test")
    await sink.publish_languages(["fr"])
    a = sink.start()
    assert sink.start() is a
    assert sink.stop() is not None
    assert sink.stop() is None
    assert sink.takes == 1


@pytest.mark.asyncio
async def test_the_tap_records_the_source_and_passes_every_frame_through(tmp_path: Path):
    sink = RecordingSink(tmp_path, "sid", "hall", "Test")
    await sink.publish_languages([])
    sink.start()
    t0 = sink.recording.started_at

    async def frames():
        for i in range(5):
            yield AudioFrame(pcm=tone(0.1, 48000), sample_rate=48000, t_audio=i * 0.1, t_wall=t0 + i * 0.1)

    seen = [f async for f in sink.tap(frames())]
    assert len(seen) == 5, "the pipeline downstream must see every frame"
    sink.stop()
    assert read(next(tmp_path.rglob("source.wav")))[1].size == 5 * int(0.1 * 48000)


@pytest.mark.asyncio
async def test_inside_a_fan_out_it_cannot_take_the_room_down(tmp_path: Path):
    """A recorder that raises must be counted, not fatal. FanOutSink's rule."""

    class Room:
        pushed = 0

        async def publish_languages(self, languages): ...
        async def push(self, language, pcm, rate): self.pushed += 1
        def queue_depth(self, language): return 0.4
        async def close(self): ...

    room = Room()
    sink = RecordingSink(tmp_path, "sid", "hall", "Test")
    fan = FanOutSink(room, sink)
    await fan.publish_languages(["fr"])
    sink.start()
    sink.recording.tracks["fr"].writer = "not a writer"  # sabotage
    await fan.push("fr", tone(0.1, 22050), 22050)
    assert room.pushed == 1
    assert fan.failures.get(0) == 1
    assert fan.queue_depth("fr") == 0.4, "a file has no playout queue; the room's depth stands"
    sink.recording.tracks["fr"].writer = None
    sink.stop()
