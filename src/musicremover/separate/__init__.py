"""Split a soundtrack into speech, music and effects.

A backend only has to separate one fixed-size chunk. `separate` feeds it a
whole soundtrack in overlapping chunks and blends them back together, so any
backend handles any length within a fixed amount of GPU memory.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

STEMS = ("speech", "music", "effects")
Stems = dict[str, np.ndarray]


class Separator(Protocol):
    sample_rate: int
    chunk_size: int  # samples per chunk

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """(channels, chunk_size) float32 -> (len(STEMS), channels, chunk_size), in STEMS order."""
        ...


def separate(
    separator: Separator,
    audio: np.ndarray,
    overlap: float = 2,
    progress: Callable[[int, int], None] | None = None,
    names: tuple[str, ...] = STEMS,
) -> Stems:
    """Separate `audio` (channels, samples) into `names`, each the same shape as `audio`.

    Chunks start every chunk_size / overlap samples (overlap may be a fraction
    such as 1.1, for a short crossfade between otherwise separate chunks).
    Each chunk is weighted by a Hann window and the sum is divided by the
    summed weights, so the joins are seamless whatever the backend returns.
    The input is zero-padded so that every real sample is covered.
    """
    if audio.ndim != 2:
        raise ValueError(f"audio must be (channels, samples), got shape {audio.shape}")
    if overlap < 1:
        raise ValueError("overlap must be at least 1")
    channels, length = audio.shape
    size = separator.chunk_size
    hop = max(1, int(size / overlap))
    pad = size - hop

    count = max(1, -(-(length + pad) // hop))  # chunks needed to cover pad + audio
    padded_length = (count - 1) * hop + size
    padded = np.zeros((channels, padded_length), dtype=np.float32)
    padded[:, pad:pad + length] = audio

    window = np.hanning(size + 2)[1:-1].astype(np.float32) if overlap > 1 else np.ones(size, np.float32)
    out = np.zeros((len(names), channels, padded_length), dtype=np.float32)
    weight = np.zeros(padded_length, dtype=np.float32)
    for k in range(count):
        start = k * hop
        estimate = separator.separate_chunk(padded[:, start:start + size])
        if estimate.shape != (len(names), channels, size):
            raise RuntimeError(f"backend returned shape {estimate.shape}, expected {(len(names), channels, size)}")
        out[:, :, start:start + size] += estimate * window
        weight[start:start + size] += window
        if progress:
            progress(k + 1, count)

    out = out[:, :, pad:pad + length] / weight[pad:pad + length]
    return {name: out[i] for i, name in enumerate(names)}
