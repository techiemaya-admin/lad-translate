"""
Audio conversion shared by the STT adapters.

Every speech recogniser in this project wants the same thing — float32 mono at
16kHz — while the room hands us int16 PCM at whatever rate the publisher
negotiated. That conversion was written three times before it was written once.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE_16K = 16_000
"""What every STT backend here expects. Not a coincidence: it is what the
acoustic front ends of Whisper, FastConformer and Qwen3-ASR were all trained
on, so resampling to it is not a choice we get to make."""


def resample_to_16k(pcm: bytes, source_rate: int) -> np.ndarray:
    """
    Convert int16 PCM to float32 mono at 16kHz.

    Linear interpolation, no anti-aliasing filter. That introduces aliasing
    when downsampling, which these models tolerate well and a proper resampler
    would avoid. Revisit if transcript quality is ever traced back to here.
    """
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if source_rate == SAMPLE_RATE_16K or samples.size == 0:
        return samples
    target_len = round(samples.size * SAMPLE_RATE_16K / source_rate)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, samples.size - 1, target_len, dtype=np.float32)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
