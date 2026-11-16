"""Voice rescue: put back laughs, shouts and gasps that Bandit puts in the music stem.

Bandit sometimes classes a non-speech voice sound (a laugh, a shout) as
music, so it is removed with the music. A second model, Demucs (htdemucs,
MIT, downloaded by Demucs itself on first use), splits the *music stem* into
voice and instruments. A laugh on its own is voice with almost no
instruments around it; a song has both. So only short frames where the voice
is far louder than the instruments are moved back. Measured in T13: a laugh
at 20:30 in the test episode has voice 31-42 dB above instruments, the
opening song never more than 13 dB.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import Stems, disk_stems, separate

FRAME_S = 0.1   # gain resolution
LOW_DB = 20.0   # voice-over-instruments ratio where rescue starts
HIGH_DB = 30.0  # ratio where the voice is moved back completely
VOICE_NAMES = ("vocals", "instruments")


_BLOCK_FRAMES = 1 << 8  # frames of float64 copies at a time (about 1.2M samples)
_BLOCK = 1 << 20        # samples per step when moving the rescued voice


def _frame_power(x: np.ndarray, frame: int) -> np.ndarray:
    """Mean power per frame of `frame` samples (last one zero padded), over channels."""
    length = x.shape[1]
    count = -(-length // frame)
    power = np.empty(count)
    for i in range(0, count, _BLOCK_FRAMES):
        a, b = i * frame, min(length, (i + _BLOCK_FRAMES) * frame)
        n = -(-(b - a) // frame)
        y = np.zeros((x.shape[0], n * frame))
        y[:, :b - a] = x[:, a:b]
        power[i:i + n] = (y ** 2).reshape(x.shape[0], n, frame).mean(axis=(0, 2))
    return power


def _frame_gains(vocals: np.ndarray, instruments: np.ndarray, sample_rate: int,
                 frame_s: float, low_db: float, high_db: float) -> tuple[np.ndarray, int]:
    """(gain per frame in 0..1, frame length in samples)."""
    if vocals.shape != instruments.shape or vocals.ndim != 2:
        raise ValueError("vocals and instruments must be 2-D arrays of the same shape")
    frame = max(1, int(frame_s * sample_rate))
    ratio = 10 * np.log10((_frame_power(vocals, frame) + 1e-12) / (_frame_power(instruments, frame) + 1e-12))
    return np.clip((ratio - low_db) / (high_db - low_db), 0.0, 1.0), frame


def _gain_block(gains: np.ndarray, frame: int, start: int, stop: int) -> np.ndarray:
    """Per-sample gain for samples [start, stop), interpolated between frame centres."""
    centres = (np.arange(len(gains)) + 0.5) * frame
    return np.interp(np.arange(start, stop), centres, gains).astype(np.float32)


def rescue_gain(vocals: np.ndarray, instruments: np.ndarray, sample_rate: int = 48000,
                frame_s: float = FRAME_S, low_db: float = LOW_DB, high_db: float = HIGH_DB) -> np.ndarray:
    """Per-sample gain (samples,) in 0..1 for the voice part of the music stem.

    0 where the voice is no more than `low_db` above the instruments, 1 from
    `high_db` up, linear in between; computed per frame and interpolated
    between frame centres so it never jumps.
    """
    gains, frame = _frame_gains(vocals, instruments, sample_rate, frame_s, low_db, high_db)
    length = vocals.shape[1]
    if not gains.any():
        return np.zeros(length, np.float32)
    out = np.empty(length, np.float32)
    for start in range(0, length, _BLOCK):
        stop = min(length, start + _BLOCK)
        out[start:stop] = _gain_block(gains, frame, start, stop)
    return out


class DemucsVoice:
    """Splits audio into (vocals, instruments) with htdemucs, as a `separate` backend."""

    sample_rate = 48000
    chunk_size = 60 * 48000

    def __init__(self, device: str = "auto") -> None:
        import torch
        from demucs.pretrained import get_model

        from .bandit import pick_device

        self.torch = torch
        self.device = pick_device(device)
        self.model = get_model("htdemucs").to(self.device).eval()
        self.vocals = self.model.sources.index("vocals")

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        import torchaudio.functional as AF
        from demucs.apply import apply_model

        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        x = AF.resample(x, self.sample_rate, self.model.samplerate)
        with torch.inference_mode():
            parts = apply_model(self.model, x[None], device=self.device, split=True, overlap=0.25,
                                progress=False)[0]
        parts = AF.resample(parts.float().cpu(), self.model.samplerate, self.sample_rate)
        parts = parts[..., :chunk.shape[1]]
        if parts.shape[-1] < chunk.shape[1]:
            parts = torch.nn.functional.pad(parts, (0, chunk.shape[1] - parts.shape[-1]))
        vocals = parts[self.vocals]
        instruments = parts.sum(0) - vocals
        return torch.stack([vocals, instruments]).numpy()


def rescue_voices(stems: Stems, voice_separator, progress: Callable[[int, int], None] | None = None,
                  out: Stems | None = None, scratch: Path | str | None = None) -> Stems:
    """Move rescued voice from the music stem to the speech stem; the stems still add up.

    The result goes to new in-memory arrays, or into `out` (float32 arrays of the same
    shape, e.g. from `disk_stems`; `out=stems` changes the stems in place). With
    `scratch`, a folder, the voice/instrument split of the music stem is kept in files
    there while it is used instead of in memory. Either way the work is done a block
    at a time, so memory use does not grow with the length of the soundtrack.
    """
    shape = stems["music"].shape
    parts = separate(voice_separator, stems["music"], overlap=1.1, progress=progress, names=VOICE_NAMES,
                     out=None if scratch is None else disk_stems(scratch, VOICE_NAMES, shape))
    gains, frame = _frame_gains(parts["vocals"], parts["instruments"], voice_separator.sample_rate,
                                FRAME_S, LOW_DB, HIGH_DB)
    result = out if out is not None else {n: np.empty(shape, np.float32) for n in ("speech", "music")}
    for a in range(0, shape[1], _BLOCK):
        b = min(shape[1], a + _BLOCK)
        rescued = parts["vocals"][:, a:b] * _gain_block(gains, frame, a, b)
        speech = stems["speech"][:, a:b] + rescued
        music = stems["music"][:, a:b] - rescued
        result["speech"][:, a:b] = speech
        result["music"][:, a:b] = music
    parts_files = [parts[n] for n in VOICE_NAMES]
    del parts
    if scratch is not None:  # free the disk space now; the folder itself is the caller's
        for part in parts_files:
            part.delete()
    return {"speech": result["speech"], "music": result["music"], "effects": stems["effects"]}
