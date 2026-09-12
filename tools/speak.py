#!/usr/bin/env python3
"""
Publish into a room as the speaker: a WAV file, or a live sound card.

The counterpart of tools/listen.py, and the headless counterpart of the
speaker page. Joins through the real join API and publishes on the source
track, so everything downstream - the session, the listeners, the output
agent - sees exactly what a phone at the lectern would produce.

A FILE, for checking a venue rig without asking anyone to talk into it for
ninety seconds:

    python tools/speak.py --room dubai-demo --base https://...run.app
    python tools/speak.py --room dubai-demo --audio fixtures/keynote.wav --loop

A SOUND CARD, which is what a real venue wants: the desk send, patched into
Dante Virtual Soundcard's input channels, published straight into the room.
No browser, and no laptop microphone picking up the room it stands in.

    python tools/speak.py --list-devices
    python tools/speak.py --room dubai-demo --base https://...run.app \\
        --device "Dante Virtual Soundcard" --channel 1

NO PROCESSING ON A CARD. The speaker page turns on echo cancellation, noise
suppression and gain control, because a phone playing a translation into the
room would otherwise feed back into its own microphone. A desk send has none
of those problems and every one of those cures hurts it: AGC pumps on a
mixed feed and noise suppression eats the tail of a sentence. Capture is raw.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import ssl
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lad_translate.obs.log import configure

FRAME_MS = 20
METER_EVERY_S = 3.0


def speaker_grant(base: str, room: str, insecure: bool) -> dict:
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(f"{base}/api/rooms/{room}/speak", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Both of these are ordinary at a venue and neither deserves a
        # traceback: one source track per room is the design, and an agent
        # started before the operator pressed APPLY is the usual order.
        if exc.code == 409:
            raise SystemExit(
                f"  someone is already speaking in {room}. A room takes one source:\n"
                "  stop the phone or the other agent first."
            ) from exc
        if exc.code == 404:
            raise SystemExit(
                f"  no live session in {room}. Start it from the console's deck (APPLY)."
            ) from exc
        raise


async def capture(args, room, grant, rtc) -> None:
    """
    Publish a live input device until interrupted.

    One channel of it: a desk send arrives on a known Dante channel and the
    room takes one source. The PortAudio callback runs on its own thread and
    does nothing but hand bytes to the loop - capture_frame is a coroutine
    and cannot be awaited there, so the frames go through a queue.
    """
    import numpy as np
    import sounddevice as sd

    from lad_translate.session.localcard import find_input_device

    card = find_input_device(args.device)
    if args.channel < 1 or args.channel > card.max_output_channels:
        print(f"  {card.name!r} has {card.max_output_channels} input channels, "
              f"not {args.channel}", file=sys.stderr)
        return
    rate = args.rate
    per_frame = int(rate * FRAME_MS / 1000)
    source = rtc.AudioSource(rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track(grant["track_name"], source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    loop = asyncio.get_running_loop()
    frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    dropped = 0
    peak = 0.0

    def on_audio(indata, count, time_info, status) -> None:
        nonlocal dropped, peak
        mono = indata[:, args.channel - 1]
        block_peak = float(np.abs(mono).max()) if mono.size else 0.0
        peak = max(peak, block_peak)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        try:
            loop.call_soon_threadsafe(frames.put_nowait, pcm)
        except (asyncio.QueueFull, RuntimeError):
            # The loop is behind or gone. Dropping is right: this is live
            # audio and a backlog only makes the room later.
            dropped += 1

    async def meter() -> None:
        """
        The level, every few seconds.

        The speaker page has a meter for this reason and the headless path
        needs one just as much: a patch that is connected but forty decibels
        down looks identical to a working one from every other angle, and
        the pipeline's VAD will treat it as silence. Measured once on a real
        rig at -37 dBFS peak, which transcribed nothing at all.
        """
        nonlocal peak
        while True:
            await asyncio.sleep(METER_EVERY_S)
            level, peak = peak, 0.0
            dbfs = 20 * math.log10(level) if level > 1e-9 else -120.0
            if dbfs < -60:
                note = "SILENT - check the Dante subscription and the channel"
            elif dbfs < -30:
                note = "very low - raise the send; the VAD may treat this as silence"
            elif dbfs > -3:
                note = "CLIPPING - lower the send"
            else:
                note = "ok"
            print(f"  input peak {dbfs:6.1f} dBFS   {note}", flush=True)

    print(f"  capturing {card.name!r} channel {args.channel} @ {rate} Hz "
          f"(no AGC, no noise suppression - it is a desk send)")
    ticker = asyncio.create_task(meter())
    with sd.InputStream(device=card.index, samplerate=rate,
                        channels=card.max_output_channels if args.channel > 1 else 1,
                        dtype="float32", blocksize=per_frame, callback=on_audio):
        try:
            while True:
                pcm = await frames.get()
                await source.capture_frame(
                    rtc.AudioFrame(data=pcm, sample_rate=rate, num_channels=1,
                                   samples_per_channel=len(pcm) // 2)
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
    if dropped:
        print(f"  {dropped} block(s) dropped: the loop could not keep up")


def monitor(args) -> int:
    """
    Levels only: no room, no publishing.

    For the half hour before a talk, when someone else may already be the
    room's one source, or when the session has not been started yet. Shows
    every channel with anything on it, so "which channel is the desk on" is
    a question the rig answers rather than a guess.
    """
    import numpy as np
    import sounddevice as sd

    from lad_translate.session.localcard import find_input_device

    card = find_input_device(args.device)
    count = min(card.max_output_channels, args.channels)
    peak = np.zeros(count)

    def on_audio(indata, frames_, time_info, status) -> None:
        np.maximum(peak, np.abs(indata[:, :count]).max(axis=0), out=peak)

    print(f"  monitoring {card.name!r} channels 1-{count} @ {args.rate} Hz. Ctrl-C to stop.")
    try:
        with sd.InputStream(device=card.index, samplerate=args.rate, channels=count,
                            dtype="float32", blocksize=480, callback=on_audio):
            while True:
                sd.sleep(int(METER_EVERY_S * 1000))
                live = [(i + 1, peak[i]) for i in range(count) if peak[i] > 0.0005]
                peak[:] = 0.0
                if not live:
                    print("  nothing on any channel", flush=True)
                    continue
                print("  " + "   ".join(
                    f"ch{ch}: {20 * math.log10(p):.0f} dBFS" for ch, p in live), flush=True)
    except KeyboardInterrupt:
        pass
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room")
    ap.add_argument("--base", help="join service base URL")
    ap.add_argument("--audio", type=Path, default=ROOT / "fixtures" / "holmes.wav")
    ap.add_argument("--loop", action="store_true", help="play the file again when it ends")
    ap.add_argument("--device", help="capture live from this input device instead of a file")
    ap.add_argument("--channel", type=int, default=1,
                    help="1-based input channel on that device (default 1)")
    ap.add_argument("--rate", type=int, default=48000, help="capture rate for --device")
    ap.add_argument("--list-devices", action="store_true",
                    help="print this machine's audio INPUT devices and exit")
    ap.add_argument("--monitor", action="store_true",
                    help="show input levels and exit; joins no room and publishes nothing")
    ap.add_argument("--channels", type=int, default=16,
                    help="how many channels --monitor watches (default 16)")
    ap.add_argument("--insecure", action="store_true", help="accept a dev TLS certificate")
    args = ap.parse_args()

    if args.list_devices:
        from lad_translate.session.localcard import list_input_devices

        devices = list_input_devices()
        if not devices:
            print("  no audio input devices on this machine")
            return 1
        print("  this machine can capture from:")
        for d in devices:
            print(f"    {d.name!r:44s} {d.max_output_channels:3d} ch  "
                  f"{int(d.default_samplerate)} Hz  ({d.hostapi})")
        print("  pass one to --device; matching is forgiving.")
        return 0

    if args.monitor:
        if not args.device:
            ap.error("--monitor needs --device")
        return monitor(args)

    if not (args.room and args.base):
        ap.error("--room and --base are required (or --list-devices)")

    configure("ERROR")
    import livekit.rtc as rtc

    grant = speaker_grant(args.base, args.room, args.insecure)
    print(f"  speaking into {args.room}  ({grant['event_name']}, {grant['source_language']})")

    room = rtc.Room()
    await room.connect(grant["url"], grant["token"])

    if args.device:
        await capture(args, room, grant, rtc)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(8)
        await room.disconnect()
        print("  done")
        return 0

    with wave.open(str(args.audio), "rb") as wav:
        rate = wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            print("  the file must be 16-bit mono", file=sys.stderr)
            return 2
        per_frame = int(rate * FRAME_MS / 1000)
        source = rtc.AudioSource(rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(grant["track_name"], source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"  publishing {args.audio.name} @ {rate} Hz" + (" on a loop" if args.loop else ""))
        try:
            while True:
                pcm = wav.readframes(per_frame)
                if not pcm:
                    if not args.loop:
                        break
                    wav.rewind()
                    continue
                await source.capture_frame(
                    rtc.AudioFrame(
                        data=pcm, sample_rate=rate, num_channels=1,
                        samples_per_channel=len(pcm) // 2,
                    )
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    # Let the tail of the pipeline drain before the source track disappears.
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(8)
    await room.disconnect()
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
