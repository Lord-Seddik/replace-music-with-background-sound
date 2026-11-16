"""T14 throwaway: per-frame spectra of the original, the stems and the output.

Usage:
    python experiments/background_scan.py OUTPUT.mkv FEATURES.npz

Reads the whole-episode stems from the T1 scan (output/bakeoff/scan/*.wav,
Bandit v2, same model as the tool) and the audio of a finished tool output,
and saves band levels and spectral flatness per 0.1 s frame for each of:
mix (original), speech, music, effects, out (tool output), residual
(mix - speech - music - effects). Analysis happens on the saved file.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SCAN = Path(__file__).resolve().parent.parent / "output" / "bakeoff" / "scan"
RATE = 48000
HOP = 4800  # 0.1 s
EDGES = [60, 250, 500, 1000, 2000, 4000, 8000, 16000]
FLAT_LO, FLAT_HI = 200, 5000


def mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE
    return audio.mean(axis=1)


def power_frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    frames = x[: n * HOP].reshape(n, HOP) * np.hanning(HOP).astype(np.float32)
    out = np.empty((n, HOP // 2 + 1), np.float32)
    for i in range(0, n, 1000):
        out[i:i + 1000] = np.abs(np.fft.rfft(frames[i:i + 1000], axis=1)) ** 2
    return out


def features(p: np.ndarray) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(HOP, 1 / RATE)
    norm = (np.hanning(HOP) ** 2).sum() * HOP / 2  # full-scale sine ~ 0 dB per band
    bands = np.stack([p[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                      for lo, hi in zip(EDGES[:-1], EDGES[1:])], axis=1)
    sel = p[:, (freqs >= FLAT_LO) & (freqs < FLAT_HI)] + 1e-20
    flat = np.exp(np.log(sel).mean(axis=1)) / sel.mean(axis=1)
    total = p[:, (freqs >= EDGES[0]) & (freqs < EDGES[-1])].sum(axis=1)
    return {"bands": 10 * np.log10(bands / norm + 1e-12),
            "level": 10 * np.log10(total / norm + 1e-12),
            "flat": flat.astype(np.float32)}


def main() -> None:
    output, dest = Path(sys.argv[1]), Path(sys.argv[2])
    sig = {name: mono(SCAN / f"{name}.wav") for name in ("mix", "speech", "music", "effects")}
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0", "-ac", "2",
                        "-ar", str(RATE), "-c:a", "pcm_f32le", str(wav)], check=True)
        sig["out"] = mono(wav)  # own downmix: ffmpeg -ac 1 adds 3 dB
    n = min(len(s) for s in sig.values())
    sig = {k: v[:n] for k, v in sig.items()}
    sig["residual"] = sig["mix"] - sig["speech"] - sig["music"] - sig["effects"]
    saved = {}
    for name, x in sig.items():
        for key, value in features(power_frames(x)).items():
            saved[f"{name}_{key}"] = value
        print(name, "done", flush=True)
    np.savez_compressed(dest, edges=np.array(EDGES), **saved)


if __name__ == "__main__":
    main()
