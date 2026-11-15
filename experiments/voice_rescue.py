"""T13 experiment: can Demucs find laughs/shouts inside Bandit's music stem?

Usage:
    python experiments/voice_rescue.py

Runs Bandit v2 on two windows (the laugh at 20:15-20:50 and the opening song,
clip 4), then Demucs htdemucs on the *music stem only*, and prints per half
second: music stem level, Demucs vocals and instruments inside it. A laugh
alone should be "voice, no instruments"; a song is "voice and instruments".
Saves the stems to output/t13/ for listening.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent.parent
EPISODE = next((ROOT / "samples").glob("*.mkv"))
OUT = ROOT / "output" / "t13"
WINDOWS = {"laugh": (1215, 35), "opening": (135, 60)}

sys.path.insert(0, str(ROOT / "src"))
from musicremover.separate import separate  # noqa: E402
from musicremover.separate.bandit import BanditV2  # noqa: E402


def load(start: float, length: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(length), "-i", str(EPISODE),
                        "-map", "0:a:0", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).T.copy()


def db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def main() -> None:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    OUT.mkdir(parents=True, exist_ok=True)
    bandit = BanditV2(ROOT / "models" / "bandit_v2_multi.ckpt")
    demucs = get_model("htdemucs").cuda().eval()
    vi = demucs.sources.index("vocals")

    for name, (start, length) in WINDOWS.items():
        mix = load(start, length)
        stems = separate(bandit, mix)
        music = torch.from_numpy(stems["music"])
        m44 = AF.resample(music, 48000, demucs.samplerate)
        with torch.inference_mode():
            parts = apply_model(demucs, m44[None], device="cuda", split=True, overlap=0.25)[0].cpu()
        parts = AF.resample(parts, demucs.samplerate, 48000)[..., :music.shape[1]].numpy()
        vocals = parts[vi]
        instruments = parts.sum(0) - vocals
        for k, v in (("mix", mix), ("music", stems["music"]), ("music_vocals", vocals),
                     ("music_instruments", instruments), ("output_now", stems["speech"] + stems["effects"])):
            sf.write(OUT / f"{name}_{k}.wav", v.T, 48000)
        np.savez(OUT / f"{name}.npz", mix=mix, vocals=vocals, instruments=instruments, **stems)

        print(f"\n== {name} (starts {start // 60}:{start % 60:02d})")
        print("  time   music  m.voc  m.inst  voc-inst")
        h = 24000
        for i in range(0, mix.shape[1] - h + 1, h):
            s = slice(i, i + h)
            t = start + i / 48000
            mv, mi = db(vocals[:, s]), db(instruments[:, s])
            print(f"  {int(t // 60)}:{t % 60:04.1f} {db(stems['music'][:, s]):6.1f} {mv:6.1f} {mi:6.1f} {mv - mi:7.1f}")


if __name__ == "__main__":
    main()
