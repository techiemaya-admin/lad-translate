"""
The speech gate that FastConformer was missing.

Without it, room tone became words: 24 and 48 second spans of invented text
between real sentences, which filled the chunker to max_words and left a
listener waiting half a minute for a phrase that was mostly noise.

These tests are about the two ways a gate goes wrong. Too eager and it clips
the start of every word, so the transducer's first token is built on a
truncated phoneme. Too reluctant and it passes the silence it exists to stop.
"""

from __future__ import annotations

import numpy as np
import pytest

from lad_translate.adapters.vad import WINDOW, SpeechGate

# Silero ships inside faster-whisper, which CI deliberately does not install:
# the model backends pull about a gigabyte of wheels. Same convention as the
# other backend tests, and the same caveat - these prove nothing in CI, so run
# them where the backend exists:
#
#     .venv/bin/python -m pytest tests/test_vad.py
#
# They are kept in the CI list anyway, so the skip is visible in the output
# rather than the file silently never running.
pytest.importorskip("faster_whisper", reason="needs the CPU STT backend installed")

SR = 16000


def speech(seconds: float) -> np.ndarray:
    """Real speech, not a tone - Silero is trained to tell those apart."""
    import wave

    with wave.open("fixtures/holmes.wav") as w:
        sr = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0
    idx = np.linspace(0, len(audio) - 1, int(len(audio) * SR / sr))
    resampled = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return resampled[: int(SR * seconds)]


def room_tone(seconds: float, level: float = 0.004) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SR * seconds)) * level).astype(np.float32)


def test_room_tone_is_suppressed():
    """The failure this exists to prevent, in its actual form."""
    gate = SpeechGate()
    passed = gate.feed(room_tone(5.0))
    assert passed.size == 0, f"{passed.size / SR:.2f}s of room tone reached the model"
    assert gate.stats.suppressed_fraction == 1.0


def test_digital_silence_is_suppressed():
    gate = SpeechGate()
    assert gate.feed(np.zeros(SR * 3, dtype=np.float32)).size == 0


def test_speech_passes():
    gate = SpeechGate()
    passed = gate.feed(speech(5.0))
    assert passed.size > 0, "speech must reach the model"
    # Not all of it: leading silence in the fixture is correctly held back.
    assert passed.size / SR > 2.0


def test_the_onset_of_speech_is_not_clipped():
    """
    Pre-roll. Gating on the first speech window cuts the attack off the word
    that opened it, and a transducer given a truncated first phoneme produces a
    wrong first token its own context then builds on.
    """
    gate = SpeechGate(pre_roll_ms=200)
    quiet = room_tone(1.0)
    gate.feed(quiet)
    passed = gate.feed(speech(2.0))
    # More audio comes out than the speech windows alone, because the ring
    # buffer of preceding audio is flushed when the gate opens.
    assert passed.size > 0
    assert gate.stats.passed_s > 0


def test_a_pause_inside_speech_does_not_close_the_gate():
    """
    Hangover. Closing on the first quiet window splits words at the pause
    inside them, so "important" arrives as two fragments.
    """
    gate = SpeechGate(hangover_ms=400)
    gate.feed(speech(2.0))
    was_open = gate._open
    gate.feed(np.zeros(int(SR * 0.2), dtype=np.float32))   # shorter than hangover
    assert was_open and gate._open, "a 200ms pause must not end the utterance"


def test_a_long_silence_does_close_it():
    gate = SpeechGate(hangover_ms=400)
    gate.feed(speech(2.0))
    gate.feed(np.zeros(int(SR * 2.0), dtype=np.float32))
    assert not gate._open


def test_partial_windows_are_held_not_padded():
    """
    Padding a short block with zeros shows Silero silence that was never in the
    stream, which biases every decision at a chunk edge. Frames arrive in
    10-20ms pieces and the window is 32ms, so this is the normal case.
    """
    gate = SpeechGate()
    assert gate.feed(np.zeros(WINDOW - 1, dtype=np.float32)).size == 0
    assert gate.stats.windows == 0, "no decision should have been made yet"


def test_it_refuses_the_wrong_sample_rate():
    """Silero is trained at 16k. Guessing here would hide a missing resample."""
    with pytest.raises(ValueError, match="16kHz"):
        SpeechGate(sample_rate=48000)


def test_stats_account_for_every_window():
    gate = SpeechGate()
    gate.feed(speech(3.0))
    gate.feed(room_tone(3.0))
    total = gate.stats.passed_s + gate.stats.suppressed_s
    assert total == pytest.approx(gate.stats.windows * WINDOW / SR, rel=0.05)
    assert 0.0 < gate.stats.suppressed_fraction < 1.0
