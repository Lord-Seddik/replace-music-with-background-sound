"""T17 helpers shared by the generation experiments (throwaway).

- SCENES: the three gap scenes cut into output/t17/scenes/ (times in the episode).
- fill_level_db: the T18b level rule (removed music - 12 dB, kept in -55..-40 dBFS).
- music_check: run the project's Bandit v2 separator on generated audio and
  report how loud its music stem is (the tool's own "music audible" limit is
  a music stem above -50 dB).
- make_listen_clip: lay generated audio under the clip's no-music track, with
  1 s fades, at the T18b level; the raw generated audio is a second track.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SCENE_DIR = ROOT / "output" / "t17" / "scenes"
BANDIT_WEIGHTS = ROOT / "models" / "bandit_v2_multi.ckpt"

# name -> (start s, length s, SigLIP label, text prompt)
SCENES = {
    "1_night_street_01-09.4": (69.4, 5.2, "night_street",
        "Quiet night ambience outdoors in a small Japanese town: crickets and night insects, "
        "a light breeze, very faint distant traffic. Field recording, no music."),
    "2_bedroom_14-12.2": (852.2, 12.0, "bedroom",
        "Quiet room tone inside a Japanese house in the daytime: soft air, faint birds and a "
        "distant street outside the window. Field recording, no music."),
    "3_park_20-51.4": (1251.4, 7.3, "park",
        "Daytime park ambience in early summer: birds, light wind in the trees, distant "
        "children playing far away. Field recording, no music."),
}


def gap_music_db(start: float) -> float:
    import csv
    for r in csv.DictReader(open(ROOT / "output" / "t15" / "gaps.csv")):
        if float(r["start"]) - 0.01 <= start < float(r["end"]):
            return float(r["music_db"])
    raise ValueError(f"no gap at {start}")


def fill_level_db(start: float) -> float:
    return float(np.clip(gap_music_db(start) - 12.0, -55.0, -40.0))


def to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """(channels, samples) float32 at sr -> 48 kHz stereo."""
    import torch
    import torchaudio.functional as F
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if sr != 48000:
        audio = F.resample(torch.from_numpy(np.ascontiguousarray(audio)), sr, 48000).numpy()
    return audio.astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(audio.astype(np.float64) ** 2) + 1e-12))


def music_check(audio48: np.ndarray, device: str = "cpu") -> dict:
    """Bandit v2 stems of the generated audio: overall and loudest-second music level."""
    from musicremover.separate.bandit import BanditV2
    sep = BanditV2(BANDIT_WEIGHTS, device)
    n = audio48.shape[1]
    size = sep.chunk_size
    stems = np.zeros((3, 2, n), np.float32)
    for a in range(0, n, size):
        chunk = np.zeros((2, size), np.float32)
        chunk[:, :min(size, n - a)] = audio48[:, a:a + size]
        stems[:, :, a:a + size] = sep.separate_chunk(chunk)[:, :, :min(size, n - a)]
    del sep
    sec = 48000
    music_1s = [rms_db(stems[1][:, i:i + sec]) for i in range(0, n - sec + 1, sec // 2)] or [rms_db(stems[1])]
    total = rms_db(audio48)
    return {
        "total_db": round(total, 1),
        "music_db": round(rms_db(stems[1]), 1),
        "music_max_1s_db": round(max(music_1s), 1),
        "effects_db": round(rms_db(stems[2]), 1),
        "speech_db": round(rms_db(stems[0]), 1),
        # relative to the generated audio itself, so it does not depend on the output level
        "music_minus_total_db": round(rms_db(stems[1]) - total, 1),
    }


def make_listen_clip(name: str, audio48: np.ndarray, out_dir: Path) -> Path:
    """Video + (no-music track + generated ambience at the T18b level) + raw generated track."""
    start, length = SCENES[name][0], SCENES[name][1]
    level = fill_level_db(start)
    amb = audio48 * 10 ** ((level - rms_db(audio48)) / 20)
    fade = np.minimum(1.0, np.minimum(np.arange(amb.shape[1]), np.arange(amb.shape[1])[::-1]) / 48000.0)
    amb = (amb * fade).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{name}.level.wav"
    sf.write(wav, amb.T, 48000, subtype="FLOAT")
    raw = out_dir / f"{name}.wav"
    out = out_dir / f"{name}_filled.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(SCENE_DIR / f"{name}.mkv"), "-i", str(wav),
        "-i", str(raw), "-filter_complex",
        "[0:a:0]aresample=48000[a0];[a0][1:a]amix=inputs=2:normalize=0:duration=first[mix]",
        "-map", "0:v:0", "-map", "[mix]", "-map", "2:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-metadata:s:a:0", f"title=No music + generated ({level:.0f} dBFS)",
        "-metadata:s:a:1", "title=Generated only (raw)", "-t", str(length), str(out)], check=True)
    return out
