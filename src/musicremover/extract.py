"""Extract an audio track to WAV with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .probe import AudioTrack


def extract(path: str | Path, track: AudioTrack, out_wav: str | Path,
            sample_rate: int = 48000) -> Path:
    """Write ``track`` as 32-bit float stereo WAV; never overwrites ``out_wav``."""
    out_wav = Path(out_wav)
    if out_wav.exists():
        raise FileExistsError(f"Output already exists: {out_wav}")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-n",
        "-i", str(path),
        "-map", f"0:a:{track.position}",
        "-vn", "-sn",
        "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out_wav


class WavArray:
    """A WAV file read as a float32 (channels, samples) array, a slice at a time.

    Supports `shape`, `ndim`, `dtype` and `a[:, i:j]` (step 1), reading only that part
    from the file, so a long soundtrack is never held in memory (and, unlike a
    memory-mapped file, never in the working set). Close it before deleting the file.
    """

    dtype = np.dtype(np.float32)
    ndim = 2

    def __init__(self, path: str | Path) -> None:
        self._file = sf.SoundFile(path)
        self.shape = (self._file.channels, self._file.frames)

    def __getitem__(self, key) -> np.ndarray:
        if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, slice) for k in key)):
            raise TypeError("WavArray supports only a[channels, start:stop] with slices")
        rows, cols = key
        if cols.step not in (None, 1):
            raise ValueError("WavArray slices cannot have a step")
        a, b, _ = cols.indices(self.shape[1])
        b = max(a, b)
        self._file.seek(a)
        data = self._file.read(b - a, dtype="float32", always_2d=True)
        if len(data) != b - a:
            raise OSError(f"{self._file.name} ended after {a + len(data)} of {self.shape[1]} frames")
        return np.ascontiguousarray(data.T)[rows]

    def __array__(self, dtype=None, copy=None):
        out = self[:, :]
        return out if dtype is None else out.astype(dtype)

    def close(self) -> None:
        self._file.close()
