"""
Session presets, each carrying the measurement that produced it.

Every number here was measured on the develop VM (n2-standard-16, CPU only) on
8 Sep 2026, and the failures are recorded alongside the successes on purpose.

The reason this file exists rather than a set of sliders: emit_interval and
max_window_s interact, and a pair that looks better on paper destroyed the
output. Moving 3.0/6.0 to 1.5/4.0 was affordable on the fixture and dropped 225
seconds of a live speaker's audio. Anyone reaching for "lower latency" at a
venue should have to read what it cost last time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    summary: str
    measured: str
    warning: str | None

    stt_backend: str
    model: str
    emit_interval: float
    window: float
    lookahead: str = "480ms"

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS: tuple[Preset, ...] = (
    Preset(
        key="live-safe",
        label="Live safe",
        summary="What a real phone in a room can actually sustain.",
        measured=(
            "whisper tiny at 3.0/6.0. tiny costs 0.26s a pass against small's "
            "1.39s, which absorbs the penalty live audio imposes."
        ),
        warning=None,
        stt_backend="faster-whisper",
        model="tiny",
        emit_interval=3.0,
        window=6.0,
    ),
    Preset(
        key="accurate",
        label="Accurate",
        summary="Better words, if the microphone is good enough.",
        measured=(
            "whisper small at 3.0/6.0. 9.4% WER against tiny's 20.1% on the "
            "fixture, and the only pair measured to hold 0% drop through a "
            "real SFU."
        ),
        warning=(
            "small cost 8.63s to decode a 4s window on a live phone against "
            "0.70s on the fixture, and dropped 72s of speech. Use it with a "
            "headset, not speakerphone, and watch the drop counter."
        ),
        stt_backend="faster-whisper",
        model="small",
        emit_interval=3.0,
        window=6.0,
    ),
    Preset(
        key="low-latency",
        label="Low latency",
        summary="Halves the delay. Measured, and it did not survive contact.",
        measured=(
            "whisper small at 1.5/4.0. On the fixture: 6.0% WER, and every "
            "timing check passed."
        ),
        warning=(
            "Live, this dropped 225 seconds of a speaker's audio. The fixture "
            "is clean read prose and does not predict a room. Do not take this "
            "to a venue without testing it in that room first."
        ),
        stt_backend="faster-whisper",
        model="small",
        emit_interval=1.5,
        window=4.0,
    ),
    Preset(
        key="streaming",
        label="Streaming (not ready)",
        summary="Sub-second on a file. Unusable in a room until it has a VAD.",
        measured=(
            "FastConformer, 480ms lookahead. On the fixture: 2.7% WER, RTF "
            "0.07, French latency p50 0.581s - better than whisper on every "
            "axis."
        ),
        warning=(
            "No voice activity detection anywhere in the adapter, so room tone "
            "becomes words: 24 and 48 second spans of text generated from "
            "near-silence. Here to be measured, not to be used."
        ),
        stt_backend="fastconformer",
        model="tiny",
        emit_interval=3.0,
        window=6.0,
    ),
)

BY_KEY = {p.key: p for p in PRESETS}
