"""T14 throwaway: try three fixes for background sound under music.

Usage:
    python experiments/tail_keeper.py START DUR OUT_PREFIX

Uses the whole-episode stems from the T1 scan (output/bakeoff/scan/*.wav)
and writes OUT_PREFIX_<variant>.wav for the window START..START+DUR (s):

  now     speech + effects (what the tool does today)
  gain24  + music at -24 dB (the existing --music-gain option)
  tails   + the noise-like part of the music stem that fits under a kept
          sound's natural decay (fast rise, ~12 dB per 0.5 s fall)
  hold    like tails, but the level is held 2 s before it falls, so a steady
          background that flips between stems stays on

Prints how much of the music stem each variant puts back (dB below it).
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, stft

SCAN = Path(__file__).resolve().parent.parent / "output" / "bakeoff" / "scan"
RATE = 48000
NFFT, HOP = 2048, 512
WARMUP = 3.0  # s read before START so envelopes are settled
FALL_DB_PER_S = 24.0  # natural decay measured without music: ~11 dB / 0.5 s
TONAL = 4.0  # a music bin this many times above its neighbours' median is a note: never put back
NEIGHBOURS = 9
CAP_DB = float(__import__("os").environ.get("CAP_DB", "-15"))  # put back at most this far below the decaying level


def read(name: str, start: float, dur: float) -> np.ndarray:
    a = int(start * RATE)
    audio, _ = sf.read(SCAN / f"{name}.wav", start=a, frames=int(dur * RATE), dtype="float32", always_2d=True)
    return audio.T


def noise_like(mag: np.ndarray) -> np.ndarray:
    """True where a bin is not a sharp spectral peak (per frame)."""
    from scipy.ndimage import median_filter
    local = median_filter(mag, size=(NEIGHBOURS, 1), mode="nearest")
    return mag <= TONAL * local


def put_back(effects: np.ndarray, music: np.ndarray, hold_s: float) -> np.ndarray:
    """Music-stem sound that fits under the kept sound's own decaying level."""
    _, _, E = stft(effects, RATE, nperseg=NFFT, noverlap=NFFT - HOP)
    _, _, M = stft(music, RATE, nperseg=NFFT, noverlap=NFFT - HOP)
    e = np.abs(E).mean(axis=0)  # (bins, frames), mid of both channels
    m = np.abs(M).mean(axis=0)
    fall = 10 ** (-FALL_DB_PER_S * HOP / RATE / 20)
    hold = int(hold_s * RATE / HOP)
    env = np.zeros_like(e)
    level = np.zeros(e.shape[0]); age = np.zeros(e.shape[0], int)
    for t in range(e.shape[1]):
        rising = e[:, t] >= level
        age = np.where(rising, 0, age + 1)
        level = np.where(rising, e[:, t], np.where(age > hold, level * fall, level))
        env[:, t] = level
    room = np.minimum(np.clip(env - e, 0, None), env * 10 ** (CAP_DB / 20))
    add = np.minimum(m, room) * noise_like(m)
    gain = np.where(m > 1e-12, add / np.maximum(m, 1e-12), 0)
    _, back = istft(M * gain[None], RATE, nperseg=NFFT, noverlap=NFFT - HOP)
    return back[:, : music.shape[1]]


def db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x ** 2) + 1e-20))


def main() -> None:
    start, dur, prefix = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    lead = min(WARMUP, start)
    s, m, e = (read(n, start - lead, dur + lead) for n in ("speech", "music", "effects"))
    cut = int(lead * RATE)
    base = s + e
    variants = {
        "now": base,
        "gain24": base + m * 10 ** (-24 / 20),
        "tails": base + put_back(e, m, 0.0),
        "hold": base + put_back(e, m, 2.0),
    }
    music_db = db(m[:, cut:])
    for name, audio in variants.items():
        audio = audio[:, cut:]
        added = audio - base[:, cut:]
        print(f"{name:7s} level {db(audio):6.1f} dB   put back {db(added) - music_db:6.1f} dB vs music stem, {db(added) - db(e[:, cut:]):6.1f} dB vs effects")
        sf.write(f"{prefix}_{name}.wav", audio.T, RATE, subtype="FLOAT")


if __name__ == "__main__":
    main()
