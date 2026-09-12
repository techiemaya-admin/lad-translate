"""
PCM plumbing shared by the hardware sinks.

A hardware sink - AES67 on the wire, a sound card in a laptop - is a ring
buffer per channel and something clocked that drains it. The rings and the
resampler are the same whichever thing drains them, so they live here and
session/aes67.py and session/localcard.py both import them.
"""

from __future__ import annotations

import threading

import numpy as np

from ..obs.log import get_logger

log = get_logger(__name__)

_soxr = None
_warned_about_soxr = False


def resample(pcm: bytes, sample_rate: int, target_rate: int) -> np.ndarray:
    """
    int16 mono PCM at one rate -> float32 at another.

    Piper renders at 22050, and 22050 -> 48000 is not an integer ratio, so
    this is a real resampler (soxr) when it is installed. The linear fallback
    exists so the module imports on a box without it, and it says so once:
    upsampling by interpolation does not alias, but it does roll off the top
    of the band, which on a handset is audible as a dull voice.
    """
    global _soxr, _warned_about_soxr
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate == target_rate or samples.size == 0:
        return samples
    if _soxr is None:
        try:
            import soxr

            _soxr = soxr
        except ImportError:
            _soxr = False
    if _soxr:
        return _soxr.resample(samples, sample_rate, target_rate, quality="HQ").astype(np.float32)
    if not _warned_about_soxr:
        _warned_about_soxr = True
        log.warning(
            "soxr is not installed; resampling by linear interpolation",
            extra={"fix": "uv pip install soxr  (or the [aes67] / [card] extra)"},
        )
    target_len = round(samples.size * target_rate / sample_rate)
    positions = np.linspace(0, samples.size - 1, target_len, dtype=np.float32)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def resampler_is_soxr() -> bool:
    global _soxr
    if _soxr is None:
        try:
            import soxr

            _soxr = soxr
        except ImportError:
            _soxr = False
    return bool(_soxr)


class Ring:
    """
    One channel's buffer between push() and the pump.

    Written from the event loop, read from the pump thread, so every access
    holds the lock. The critical sections are a few hundred samples; the pump
    holds it for well under the millisecond it has.
    """

    __slots__ = ("buf", "fill", "lock", "rate", "read_at")

    def __init__(self, capacity: int, rate: int) -> None:
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.rate = rate
        self.read_at = 0
        self.fill = 0
        self.lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self.buf.size

    def free(self) -> int:
        with self.lock:
            return self.capacity - self.fill

    def seconds(self) -> float:
        with self.lock:
            return self.fill / self.rate

    def write(self, samples: np.ndarray) -> int:
        """Append what fits. Returns how many samples were written."""
        with self.lock:
            n = min(samples.size, self.capacity - self.fill)
            if n <= 0:
                return 0
            write_at = (self.read_at + self.fill) % self.capacity
            first = min(n, self.capacity - write_at)
            self.buf[write_at : write_at + first] = samples[:first]
            if n > first:
                self.buf[: n - first] = samples[first:n]
            self.fill += n
            return n

    def read(self, n: int, out: np.ndarray) -> int:
        """Fill `out` (length n) with buffered audio, zeros past the end.
        Returns how many real samples there were."""
        with self.lock:
            have = min(n, self.fill)
            first = min(have, self.capacity - self.read_at)
            out[:first] = self.buf[self.read_at : self.read_at + first]
            if have > first:
                out[first:have] = self.buf[: have - first]
            if have < n:
                out[have:n] = 0.0
            self.read_at = (self.read_at + have) % self.capacity
            self.fill -= have
            return have
