"""Split a soundtrack into speech, music and effects.

A backend only has to separate one fixed-size chunk. `separate` feeds it a
whole soundtrack in overlapping chunks and blends them back together, so any
backend handles any length within a fixed amount of GPU memory.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
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


class FileArray:
    """A float32 (channels, samples) array that lives in a plain file and is read and
    written with ordinary file calls, a slice at a time.

    Unlike a memory-mapped file, nothing stays in the process's working set, so memory
    use does not grow with the length of the soundtrack. Supports what the pipeline
    uses: `shape`, `ndim`, `dtype`, `a[:, i:j]` and `a[:, i:j] = x` (also `+=`).
    Each channel is stored whole, one after the other, in a raw file. `np.asarray(a)`
    reads everything into memory: only for small arrays (tests).
    """

    dtype = np.dtype(np.float32)
    ndim = 2

    def __init__(self, path: Path | str, shape: tuple[int, int]) -> None:
        self.path = Path(path)
        self.shape = (int(shape[0]), int(shape[1]))
        self._file = open(self.path, "w+b")
        self._file.truncate(self.shape[0] * self.shape[1] * 4)

    def _span(self, key) -> tuple[range, int, int]:
        if isinstance(key, slice):  # a[rows] means a[rows, :]
            key = (key, slice(None))
        if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, slice) for k in key)):
            raise TypeError("FileArray supports only a[channels, start:stop] with slices")
        rows, cols = key
        if cols.step not in (None, 1):
            raise ValueError("FileArray slices cannot have a step")
        a, b, _ = cols.indices(self.shape[1])
        return range(*rows.indices(self.shape[0])), a, max(a, b)

    def __getitem__(self, key) -> np.ndarray:
        rows, a, b = self._span(key)
        out = np.empty((len(rows), b - a), np.float32)
        for i, ch in enumerate(rows):
            self._file.seek((ch * self.shape[1] + a) * 4)
            if self._file.readinto(out[i]) != (b - a) * 4:
                raise OSError(f"{self.path} is shorter than expected")
        return out

    def __setitem__(self, key, value) -> None:
        rows, a, b = self._span(key)
        data = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(rows), b - a))
        for i, ch in enumerate(rows):
            self._file.seek((ch * self.shape[1] + a) * 4)
            self._file.write(np.ascontiguousarray(data[i]))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self[:, :]
        return out if dtype is None else out.astype(dtype)

    def close(self) -> None:
        self._file.close()

    def delete(self) -> None:
        """Close the file and remove it, freeing its disk space now."""
        self.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def disk_stems(directory: Path | str, names: tuple[str, ...], shape: tuple[int, int]) -> Stems:
    """Empty float32 stems of `shape` that live in files in `directory`, not in memory.

    Pass the result as `out=` to `separate` so a long soundtrack's stems are
    written as they are finished. The caller removes the directory afterwards
    (on Windows only once the arrays are no longer referenced).
    """
    return {name: FileArray(Path(directory) / f"{name}.f32", shape) for name in names}


def separate(
    separator: Separator,
    audio: np.ndarray,
    overlap: float = 2,
    progress: Callable[[int, int], None] | None = None,
    names: tuple[str, ...] = STEMS,
    out: Stems | None = None,
) -> Stems:
    """Separate `audio` (channels, samples) into `names`, each the same shape as `audio`.

    Chunks start every chunk_size / overlap samples (overlap may be a fraction
    such as 1.1, for a short crossfade between otherwise separate chunks).
    Each chunk is weighted by a Hann window and the sum is divided by the
    summed weights, so the joins are seamless whatever the backend returns.
    The input is zero-padded so that every real sample is covered.

    A sample is final as soon as the next chunk starts after it, so only about
    one chunk of blending buffers is held, not the whole soundtrack; finished
    samples are written to `out` (one float32 array of the audio's shape per
    name, e.g. from `disk_stems`) or, by default, to new arrays in memory.
    """
    if audio.ndim != 2:
        raise ValueError(f"audio must be (channels, samples), got shape {audio.shape}")
    if overlap < 1:
        raise ValueError("overlap must be at least 1")
    channels, length = audio.shape
    size = separator.chunk_size
    hop = max(1, int(size / overlap))
    pad = size - hop
    if out is None:
        out = {name: np.zeros((channels, length), dtype=np.float32) for name in names}
    elif set(out) != set(names) or any(a.shape != audio.shape or a.dtype != np.float32 for a in out.values()):
        raise ValueError(f"out must hold float32 arrays of shape {audio.shape} named {names}")

    count = max(1, -(-(length + pad) // hop))  # chunks needed to cover pad + audio
    window = np.hanning(size + 2)[1:-1].astype(np.float32) if overlap > 1 else np.ones(size, np.float32)
    # Blending buffers for the padded range [start, start + size) of the current chunk.
    acc = np.zeros((len(names), channels, size), dtype=np.float32)
    weight = np.zeros(size, dtype=np.float32)
    chunk = np.zeros((channels, size), dtype=np.float32)
    for k in range(count):
        start = k * hop  # position in the padded signal; real sample i is at i + pad
        lo, hi = max(start, pad), min(start + size, pad + length)
        chunk[:] = 0
        if hi > lo:
            chunk[:, lo - start:hi - start] = audio[:, lo - pad:hi - pad]
        estimate = separator.separate_chunk(chunk)
        if estimate.shape != (len(names), channels, size):
            raise RuntimeError(f"backend returned shape {estimate.shape}, expected {(len(names), channels, size)}")
        acc += estimate * window
        weight += window

        done = size if k == count - 1 else hop  # samples no later chunk reaches
        lo, hi = max(start, pad), min(start + done, pad + length)
        if hi > lo:
            for i, name in enumerate(names):
                out[name][:, lo - pad:hi - pad] = acc[i, :, lo - start:hi - start] / weight[lo - start:hi - start]
        if k < count - 1:  # slide the buffers so index 0 is the next chunk's start
            acc[:, :, :size - done] = acc[:, :, done:]
            acc[:, :, size - done:] = 0
            weight[:size - done] = weight[done:]
            weight[size - done:] = 0
        if progress:
            progress(k + 1, count)
    return out
