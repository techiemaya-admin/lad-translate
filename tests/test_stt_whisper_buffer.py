"""
The STT buffer must honour its own window cap.

Two thresholds guard the same audio and they do not meet:

    _track_silence      counts quiet below silence_rms (0.005)
    _transcribe_buffer  refuses to transcribe below speech_rms (0.006)

Audio between them is not quiet enough to finalise and too quiet to
transcribe. It used to hit `continue` with the buffer intact, so max_window_s
stopped being enforced and the buffer grew for as long as the band held.

That is not a slow leak. Every later pass transcribes a longer window, the
room-to-STT queue backs up behind it, and the backpressure guard shreds live
audio to catch up. On the develop box it produced a 7.11s buffer against a 6.0s
cap, 52 seconds of speech dropped, and a transcript of disconnected fragments -
while the machine sat at load 3.8 of 16 cores, which is why it read as anything
except a capacity problem.

A phone microphone in a quiet room sits in that band, so this is the ordinary
case rather than an exotic one.
"""

from __future__ import annotations

import numpy as np
import pytest

from lad_translate.adapters.base import AudioFrame
from lad_translate.adapters.stt_whisper import WhisperSttAdapter

SAMPLE_RATE = 16000
FRAME_S = 0.01


def frames_at_rms(rms: float, count: int):
    """A stream of frames whose amplitude is exactly `rms`."""
    async def gen():
        n = int(SAMPLE_RATE * FRAME_S)
        pcm = (np.full(n, rms, dtype=np.float32) * 32768.0).astype(np.int16).tobytes()
        for i in range(count):
            yield AudioFrame(pcm, SAMPLE_RATE, i * FRAME_S, i * FRAME_S)
    return gen()


@pytest.mark.asyncio
async def test_window_cap_holds_for_audio_between_the_thresholds():
    """
    The exact failure: loud enough to not be silence, quiet enough to not be
    speech. Before the fix the buffer grew past max_window_s without bound.
    """
    adapter = WhisperSttAdapter(max_window_s=6.0, emit_interval=3.0)
    # A sentinel, not a model: transcribe() refuses to run without one, and
    # _transcribe_buffer is stubbed out so it is never consulted. The branch
    # under test is the one that runs when the model returns nothing.
    adapter._model = object()
    adapter._transcribe_buffer = _never_transcribes  # type: ignore[method-assign]

    dead_band = (adapter.silence_rms + adapter.speech_rms) / 2
    assert adapter.silence_rms < dead_band < adapter.speech_rms, "not in the band"

    # 30 seconds of it - five times the cap.
    async for _ in adapter.transcribe(frames_at_rms(dead_band, int(30 / FRAME_S))):
        pass

    held_s = adapter._buffer.size / SAMPLE_RATE
    assert held_s <= adapter.max_window_s, (
        f"buffer grew to {held_s:.2f}s against a {adapter.max_window_s}s cap; "
        "the window bound is not being enforced"
    )


@pytest.mark.asyncio
async def test_silence_below_both_thresholds_still_clears():
    """The uncontested case, so the fix is not mistaken for the whole story."""
    adapter = WhisperSttAdapter(max_window_s=6.0, emit_interval=3.0)
    adapter._model = object()
    adapter._transcribe_buffer = _never_transcribes  # type: ignore[method-assign]

    async for _ in adapter.transcribe(frames_at_rms(0.0, int(30 / FRAME_S))):
        pass

    held_s = adapter._buffer.size / SAMPLE_RATE
    assert held_s <= adapter.max_window_s


async def _never_transcribes() -> str:
    """What faster-whisper effectively does below speech_rms: nothing."""
    return ""
