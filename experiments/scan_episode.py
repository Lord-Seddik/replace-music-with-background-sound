"""T1 throwaway: separate a whole episode with Bandit v2 and log stem levels.

Usage:
    python experiments/scan_episode.py samples/episode.mkv

Writes output/bakeoff/scan/{speech,music,effects}.wav, levels.csv (dB per
second per stem) and prints the 60 s windows with the most music and the
most effects. Also reports run time for the full episode.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff as b  # noqa: E402

OUT = b.OUT / "scan"
WINDOW = 60


def db(x: np.ndarray) -> np.ndarray:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def best_window(levels: np.ndarray, skip: set[int]) -> int:
    power = 10 ** (levels / 10)
    sums = np.convolve(power, np.ones(WINDOW), mode="valid")
    for start in np.argsort(sums)[::-1]:
        if all(abs(start - s) >= WINDOW for s in skip):
            return int(start)
    return 0


def main() -> None:
    episode = Path(sys.argv[1]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "mix.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(episode), "-map", "0:a:0", "-ac", "2",
                    "-ar", "48000", "-c:a", "pcm_f32le", str(wav)], check=True)
    mix, rate = sf.read(wav, dtype="float32", always_2d=True)
    mix = mix.T.copy()

    model, config = b.get_model_from_config("bandit_v2", str(b.MODELS / "config_bandit_v2.yaml"))
    b.load_weights(model, b.MODELS / "bandit_v2_multi.ckpt")
    model = model.cuda().eval()
    config.inference.batch_size = 1
    config.inference.num_overlap = 2

    start = time.perf_counter()
    # 5-minute pieces: the runner keeps a whole input and its stems on the GPU,
    # which does not fit in 4 GB for a full episode. Seams are fine for a level scan.
    piece = 300 * rate
    parts = []
    with torch.inference_mode():
        for i in range(0, mix.shape[1], piece):
            parts.append(b.demix(config, model, mix[:, i:i + piece], torch.device("cuda"), model_type="bandit_v2"))
            torch.cuda.empty_cache()
            print(f"  {min(i + piece, mix.shape[1]) / rate / 60:.0f} min done", flush=True)
    stems = {k: np.concatenate([p[k] for p in parts], axis=1) for k in parts[0]}
    seconds = time.perf_counter() - start
    length = mix.shape[1] / rate
    print(f"episode {length / 60:.1f} min separated in {seconds / 60:.1f} min ({length / seconds:.2f}x realtime)")

    names = {"speech": "speech", "music": "music", "sfx": "effects"}
    per_second = {}
    for key, name in names.items():
        audio = stems[key]
        b.write(OUT / f"{name}.wav", audio, rate)
        n = audio.shape[1] // rate
        per_second[name] = np.array([db(audio[:, i * rate:(i + 1) * rate].T) for i in range(n)])

    with open(OUT / "levels.csv", "w") as f:
        f.write("second,speech_db,music_db,effects_db\n")
        for i in range(len(per_second["speech"])):
            f.write(f"{i},{per_second['speech'][i]:.1f},{per_second['music'][i]:.1f},{per_second['effects'][i]:.1f}\n")

    chosen: set[int] = set()
    for name in ("music", "effects"):
        s = best_window(per_second[name], chosen)
        chosen.add(s)
        print(f"most {name}: {s // 60}:{s % 60:02d}")


if __name__ == "__main__":
    main()
