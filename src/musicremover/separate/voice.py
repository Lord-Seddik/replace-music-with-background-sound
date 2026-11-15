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

import numpy as np

from . import Stems, separate

FRAME_S = 0.1   # gain resolution
LOW_DB = 20.0   # voice-over-instruments ratio where rescue starts
HIGH_DB = 30.0  # ratio where the voice is moved back completely
VOICE_NAMES = ("vocals", "instruments")


def rescue_gain(vocals: np.ndarray, instruments: np.ndarray, sample_rate: int = 48000,
                frame_s: float = FRAME_S, low_db: float = LOW_DB, high_db: float = HIGH_DB) -> np.ndarray:
    """Per-sample gain (samples,) in 0..1 for the voice part of the music stem.

    0 where the voice is no more than `low_db` above the instruments, 1 from
    `high_db` up, linear in between; computed per frame and interpolated
    between frame centres so it never jumps.
    """
    if vocals.shape != instruments.shape or vocals.ndim != 2:
        raise ValueError("vocals and instruments must be 2-D arrays of the same shape")
    length = vocals.shape[1]
    frame = max(1, int(frame_s * sample_rate))
    count = -(-length // frame)
    padded = count * frame

    def power(x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0], padded), np.float32)
        y[:, :length] = x
        return (y.astype(np.float64) ** 2).reshape(x.shape[0], count, frame).mean(axis=(0, 2))

    ratio = 10 * np.log10((power(vocals) + 1e-12) / (power(instruments) + 1e-12))
    gains = np.clip((ratio - low_db) / (high_db - low_db), 0.0, 1.0)
    if not gains.any():
        return np.zeros(length, np.float32)
    centres = (np.arange(count) + 0.5) * frame
    out = np.empty(length, np.float32)
    block = 1 << 22
    for start in range(0, length, block):
        stop = min(length, start + block)
        out[start:stop] = np.interp(np.arange(start, stop), centres, gains)
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


def rescue_voices(stems: Stems, voice_separator, progress: Callable[[int, int], None] | None = None) -> Stems:
    """Move rescued voice from the music stem to the speech stem; the stems still add up."""
    parts = separate(voice_separator, stems["music"], overlap=1.1, progress=progress, names=VOICE_NAMES)
    gain = rescue_gain(parts["vocals"], parts["instruments"], voice_separator.sample_rate)
    rescued = parts["vocals"] * gain
    return {
        "speech": stems["speech"] + rescued,
        "music": stems["music"] - rescued,
        "effects": stems["effects"],
    }
