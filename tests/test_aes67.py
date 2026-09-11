"""
The AES67 sink, judged by what reaches a receiver.

Every test here opens a UDP socket, points the sink at it, and decodes the
packets - because a clocked sender has failure modes that no amount of
inspecting its internals reveals: a pump that runs at 900 packets a second
instead of 1000, a channel that comes out on the wrong wire, a gain applied
to the wrong language. A receiver is the only vantage point that sees them.

Unicast to 127.0.0.1 rather than multicast, so the tests run on a CI box
whose loopback interface does not route multicast. The sink does not care;
sendto() is sendto().

What these cannot tell you: whether a Dante device in AES67 mode accepts the
flow. That needs a Dante device, a PTP grandmaster, and a room. See the module
docstring in session/aes67.py.
"""

from __future__ import annotations

import asyncio
import socket
import struct
import time

import numpy as np
import pytest

from lad_translate.config import OutputChannel, OutputDevice
from lad_translate.session import aes67
from lad_translate.session.aes67 import (
    FRAMES_PER_PACKET,
    RATE,
    Aes67Config,
    Aes67Sink,
    pack_l24,
    to_48k,
    unpack_l24,
)
from lad_translate.session.sinks import AudioSink


def a_device(channels: tuple[OutputChannel, ...], channel_count: int = 16, **over) -> OutputDevice:
    fields = {
        "device_id": "d1",
        "name": "Main hall",
        "device_name": "AES67",
        "kind": "aes67",
        "channel_count": channel_count,
        "sample_rate": RATE,
        "channels": channels,
    }
    fields.update(over)
    return OutputDevice(**fields)


class Receiver:
    """A UDP listener that keeps every packet, decoded."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]

    def config(self, **over) -> Aes67Config:
        return Aes67Config(multicast_base="127.0.0.1", port=self.port, sap=False, **over)

    def drain(self, min_packets: int = 1, seconds: float = 1.0) -> list[dict]:
        """Collect for `seconds` (or until min_packets after that), decoded."""
        packets: list[dict] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline or len(packets) < min_packets:
            try:
                data, _ = self.sock.recvfrom(65536)
            except TimeoutError:
                if time.monotonic() > deadline + 2.0:
                    break
                continue
            v_p_x_cc, m_pt, seq, ts, ssrc = struct.unpack("!BBHII", data[:12])
            packets.append(
                {
                    "version": v_p_x_cc >> 6,
                    "pt": m_pt & 0x7F,
                    "marker": bool(m_pt & 0x80),
                    "seq": seq,
                    "ts": ts,
                    "ssrc": ssrc,
                    "payload": data[12:],
                }
            )
        return packets

    def close(self) -> None:
        self.sock.close()


def tone(seconds: float, hz: float = 440.0, rate: int = 22050, amp: float = 0.5) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype(np.int16).tobytes()


# --- the codec ---------------------------------------------------------------


def test_l24_round_trips_and_is_big_endian():
    frames = np.array([[0.5, -0.5], [1.0, -1.0], [0.0, 0.25]], dtype=np.float32)
    packed = pack_l24(frames)
    assert len(packed) == 3 * 2 * 3
    # +0.5 full scale is 0x3FFFFF-ish; its first byte on the wire is 0x3F.
    assert packed[0] == 0x3F
    # -0.5 is two's complement: first byte 0xC0.
    assert packed[3] == 0xC0
    back = unpack_l24(packed, channels=2)
    assert np.allclose(back, frames, atol=2e-7)


def test_resampling_keeps_the_pitch():
    """22050 -> 48000 is not an integer ratio; a real resampler keeps 440 Hz at 440 Hz."""
    out = to_48k(tone(1.0, hz=440.0), 22050)
    assert abs(out.size - RATE) <= 2
    spectrum = np.abs(np.fft.rfft(out[:RATE]))
    peak_hz = np.argmax(spectrum[1:]) + 1  # bins are 1 Hz apart over one second
    assert abs(peak_hz - 440) <= 1


def test_already_48k_is_passed_through_untouched():
    pcm = tone(0.1, rate=RATE)
    out = to_48k(pcm, RATE)
    assert out.size == len(pcm) // 2


# --- the shape of the sink ---------------------------------------------------


def test_it_is_an_audio_sink():
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)))
    assert isinstance(sink, AudioSink)


def test_a_sixteen_channel_card_with_ten_patched_is_two_trimmed_flows():
    """
    Channels 1-8 and 9-16 would be two eight-channel flows; with channels
    1, 2 and 10 patched they are a 2-channel flow and a 2-channel flow, on
    consecutive groups. Nine megabits of silence per empty flow is the reason.
    """
    device = a_device(
        (OutputChannel("en", 1), OutputChannel("fr", 2), OutputChannel("ar", 10)),
    )
    sink = Aes67Sink(device, Aes67Config(multicast_base="239.69.5.1", sap=False))
    flows = sink._build_flows()
    assert [(f.group, f.channel_count) for f in flows] == [("239.69.5.1", 2), ("239.69.5.2", 2)]
    # Channel 10 is position 2 of flow 2 (channel 9 is a silent gap, kept so
    # the receiver's numbering stays the device's).
    assert flows[1].rings[0] is None and flows[1].rings[1] is not None


def test_a_card_with_only_high_channels_patched_sends_only_that_flow():
    device = a_device((OutputChannel("fr", 12),))
    sink = Aes67Sink(device, Aes67Config(sap=False))
    flows = sink._build_flows()
    assert len(flows) == 1
    assert flows[0].channel_count == 4  # channels 9..12


@pytest.mark.asyncio
async def test_a_disabled_device_is_refused_at_open():
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),), enabled=False))
    with pytest.raises(RuntimeError, match="disabled"):
        await sink.publish_languages(["fr"])


@pytest.mark.asyncio
async def test_a_44100_profile_is_refused_rather_than_resampled_to_it():
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),), sample_rate=44100))
    with pytest.raises(RuntimeError, match="48000"):
        await sink.publish_languages(["fr"])


# --- on the wire --------------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_flows_before_anyone_speaks():
    """
    The whole reason this is a pump and not a wrapper. A receiver that gets
    nothing between phrases does not hear silence; it drops the flow.
    """
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1), OutputChannel("ar", 2))), rx.config())
    await sink.publish_languages(["fr", "ar"])
    try:
        packets = rx.drain(min_packets=50, seconds=0.2)
    finally:
        await sink.close()
        rx.close()

    assert len(packets) >= 50
    p = packets[0]
    assert p["version"] == 2
    assert p["pt"] == 98
    assert len(p["payload"]) == FRAMES_PER_PACKET * 2 * 3
    assert not np.any(unpack_l24(p["payload"], 2))


@pytest.mark.asyncio
async def test_sequence_and_timestamp_advance_by_one_and_forty_eight():
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)), rx.config())
    await sink.publish_languages(["fr"])
    try:
        packets = rx.drain(min_packets=100, seconds=0.15)
    finally:
        await sink.close()
        rx.close()

    seqs = [p["seq"] for p in packets]
    tss = [p["ts"] for p in packets]
    ssrcs = {p["ssrc"] for p in packets}
    assert len(ssrcs) == 1
    seq_steps = {(b - a) & 0xFFFF for a, b in zip(seqs, seqs[1:])}
    ts_steps = {(b - a) & 0xFFFFFFFF for a, b in zip(tss, tss[1:])}
    # A late catch-up would show as a jump in both, by the same multiple.
    assert seq_steps <= {1} or all(
        (s * FRAMES_PER_PACKET) in ts_steps for s in seq_steps
    ), (seq_steps, ts_steps)
    assert all(step % FRAMES_PER_PACKET == 0 for step in ts_steps)


@pytest.mark.asyncio
async def test_the_pump_runs_at_a_thousand_packets_a_second():
    """
    Loose on purpose: a CI box under load will not hold 1 ms, and the pump
    skips ahead rather than bursting when it falls behind. What must hold is
    the *timestamp* rate - the media clock - not the packet count.
    """
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)), rx.config())
    await sink.publish_languages(["fr"])
    try:
        started = time.monotonic()
        packets = rx.drain(min_packets=200, seconds=0.5)
        elapsed = time.monotonic() - started
    finally:
        await sink.close()
        rx.close()

    ts_span = (packets[-1]["ts"] - packets[0]["ts"]) & 0xFFFFFFFF
    media_seconds = ts_span / RATE
    # Media time advanced at wall-clock rate, within 15% over half a second.
    assert abs(media_seconds - elapsed) < 0.15 * elapsed + 0.02, (media_seconds, elapsed)
    assert len(packets) > 200


@pytest.mark.asyncio
async def test_speech_lands_on_its_own_channels_with_its_own_gain():
    """
    French on channel 1 at 0 dB and again on channel 3 at -6 dB; Arabic on
    channel 2. Push a French tone and nothing else: channel 2 must stay
    silent, channel 3 must be half of channel 1.
    """
    rx = Receiver()
    device = a_device(
        (
            OutputChannel("fr", 1),
            OutputChannel("ar", 2),
            OutputChannel("fr", 3, gain_db=-6.0),
        )
    )
    sink = Aes67Sink(device, rx.config())
    await sink.publish_languages(["fr", "ar"])
    try:
        await sink.push("fr", tone(0.3, hz=1000.0), 22050)
        await asyncio.sleep(0.05)
        packets = rx.drain(min_packets=150, seconds=0.25)
    finally:
        await sink.close()
        rx.close()

    audio = np.concatenate([unpack_l24(p["payload"], 3) for p in packets])
    rms = np.sqrt(np.mean(audio**2, axis=0))
    assert rms[0] > 0.05, "French did not reach channel 1"
    assert rms[1] < 1e-4, "Arabic's channel carried audio nobody pushed"
    assert abs(rms[2] / rms[0] - 10 ** (-6 / 20)) < 0.05, "channel 3 is not 6 dB down"


@pytest.mark.asyncio
async def test_a_language_the_rig_does_not_carry_is_ignored_not_fatal():
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)), rx.config())
    await sink.publish_languages(["fr", "de"])
    try:
        await sink.push("de", tone(0.1), 22050)  # nothing mapped for de
        assert sink.queue_depth("de") == 0.0
        assert sink.stats.pushes == 0
    finally:
        await sink.close()
        rx.close()


@pytest.mark.asyncio
async def test_queue_depth_is_what_was_pushed_and_not_yet_sent():
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)), rx.config())
    await sink.publish_languages(["fr"])
    try:
        await sink.push("fr", tone(1.0), 22050)
        depth = sink.queue_depth("fr")
        assert 0.85 < depth <= 1.0, depth
        await asyncio.sleep(0.3)
        later = sink.queue_depth("fr")
        assert later < depth - 0.2, (depth, later)
    finally:
        await sink.close()
        rx.close()


@pytest.mark.asyncio
async def test_a_full_ring_waits_then_drops_loudly(caplog):
    """
    Two seconds of buffer, four seconds of speech pushed at once. The second
    push cannot fit; rather than overwrite what the pump has not sent, push()
    waits for space up to the timeout and then drops - counted and logged,
    because a phrase that silently vanished is the one failure a venue post
    mortem cannot reconstruct.
    """
    import logging

    caplog.set_level(logging.ERROR)
    aes67.PUSH_WAIT_TIMEOUT_S, saved = 0.2, aes67.PUSH_WAIT_TIMEOUT_S
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 1),)), rx.config())
    await sink.publish_languages(["fr"])
    try:
        await sink.push("fr", tone(1.9), 22050)
        await sink.push("fr", tone(1.9), 22050)
        assert sink.stats.dropped_phrases == 1
        assert sink.stats.dropped_seconds == pytest.approx(1.9, abs=0.05)
        assert any("ring full" in r.getMessage() for r in caplog.records)
    finally:
        aes67.PUSH_WAIT_TIMEOUT_S = saved
        await sink.close()
        rx.close()


# --- announcement ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sdp_says_what_a_dante_receiver_needs_to_hear():
    rx = Receiver()
    device = a_device((OutputChannel("fr", 1, ir_channel=1), OutputChannel("ar", 2, ir_channel=2)))
    sink = Aes67Sink(
        device,
        rx.config(ptp_grandmaster="00-1d-c1-ff-fe-12-34-56", ptp_domain=0),
    )
    await sink.publish_languages(["fr", "ar"])
    try:
        flow = sink._flows[0]
        sdp = sink.sdp(flow)
        packet = sink.sap_packet(flow)
    finally:
        await sink.close()
        rx.close()

    assert f"m=audio {rx.port} RTP/AVP 98" in sdp
    assert "a=rtpmap:98 L24/48000/2" in sdp
    assert "a=ptime:1" in sdp
    assert "a=ts-refclk:ptp=IEEE1588-2008:00-1d-c1-ff-fe-12-34-56:0" in sdp
    assert "a=mediaclk:direct=0" in sdp
    assert "c=IN IP4 127.0.0.1/16" in sdp

    # SAP: version 1, announcement, IPv4 origin, then the payload type and SDP.
    assert packet[0] == 0x20
    assert b"application/sdp\x00v=0" in packet
    deletion = sink.sap_packet(flow, deletion=True)
    assert deletion[0] == 0x24


@pytest.mark.asyncio
async def test_flows_report_what_is_on_the_wire():
    rx = Receiver()
    sink = Aes67Sink(a_device((OutputChannel("fr", 2, ir_channel=1),)), rx.config())
    await sink.publish_languages(["fr"])
    try:
        assert sink.flows == [
            {
                "group": "127.0.0.1",
                "port": rx.port,
                "channels": 2,
                "carried": [{"channel": 2, "language": "fr", "ir_channel": 1}],
            }
        ]
    finally:
        await sink.close()
        rx.close()
