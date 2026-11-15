"""T1 throwaway: run each candidate model on the test clips.

Usage:
    python experiments/bakeoff.py [model ...] [clipN ...]   # default: all models, all clips

For each clip and model it writes, under output/bakeoff/<model>/<clip>/:
    speech.wav, music.wav, effects.wav   the separated stems
    nomusic.wav                          speech + effects, what the tool would output
and appends speed and GPU memory to output/bakeoff/results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent.parent
MSST = ROOT / "experiments" / "msst"
MODELS = ROOT / "models"
CLIPS = ROOT / "output" / "bakeoff" / "clips"
OUT = ROOT / "output" / "bakeoff"
CLIP_RATE = 48000

sys.path.insert(0, str(MSST))
from utils.settings import get_model_from_config  # noqa: E402
from utils.model_utils import demix  # noqa: E402

CANDIDATES = {
    # name: (msst model type, config, checkpoint, stem names -> speech/music/effects)
    "bandit_v2_multi": ("bandit_v2", "config_bandit_v2.yaml", "bandit_v2_multi.ckpt",
                        {"speech": "speech", "music": "music", "sfx": "effects"}),
    "bandit_plus": ("bandit", "config_bandit_plus.yaml", "bandit_plus_dnr.chpt",
                    {"speech": "speech", "music": "music", "effects": "effects"}),
}


def load_weights(model: torch.nn.Module, path: Path) -> None:
    # weights_only: refuse checkpoints that carry code, not just tensors
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "state", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    wanted = set(model.state_dict())
    for prefix in ("", "model.", "_orig_mod.", "model._orig_mod."):
        trial = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if wanted <= set(trial):
            model.load_state_dict({k: trial[k] for k in wanted})
            return
    missing = sorted(wanted - set(state))[:5]
    raise RuntimeError(f"checkpoint keys do not match the model, e.g. missing {missing}")


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    return AF.resample(torch.from_numpy(audio), src, dst).numpy()


def write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, rate, subtype="FLOAT")


def run_msst(name: str, clips: list[Path], device: torch.device) -> list[dict]:
    model_type, config_name, ckpt_name, stem_map = CANDIDATES[name]
    model, config = get_model_from_config(model_type, str(MODELS / config_name))
    load_weights(model, MODELS / ckpt_name)
    model = model.to(device).eval()
    config.inference.batch_size = 1  # fit in 4 GB
    rate = config.audio.sample_rate

    rows = []
    for clip in clips:
        mix, _ = sf.read(clip, dtype="float32", always_2d=True)
        mix = resample(mix.T.copy(), CLIP_RATE, rate)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.inference_mode():
            stems = demix(config, model, mix, device, model_type=model_type)
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0

        named = {stem_map[k]: resample(v.astype(np.float32), rate, CLIP_RATE) for k, v in stems.items()}
        folder = OUT / name / clip.stem
        for stem, audio in named.items():
            write(folder / f"{stem}.wav", audio, CLIP_RATE)
        write(folder / "nomusic.wav", named["speech"] + named["effects"], CLIP_RATE)
        rows.append(result_row(name, clip, mix.shape[1] / rate, seconds, peak, device))
    return rows


def run_demucs(clips: list[Path], device: torch.device) -> list[dict]:
    """Baseline: an ordinary music model. Its 'vocals' stem is the only one without
    instruments, so it is what 'no music' means for it. Effects are lost."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs").to(device).eval()
    rows = []
    for clip in clips:
        mix, _ = sf.read(clip, dtype="float32", always_2d=True)
        wav = resample(mix.T.copy(), CLIP_RATE, model.samplerate)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.inference_mode():
            out = apply_model(model, torch.from_numpy(wav)[None], device=device, split=True, overlap=0.25)[0]
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0

        folder = OUT / "demucs_baseline" / clip.stem
        stems = {s: resample(out[i].cpu().numpy(), model.samplerate, CLIP_RATE) for i, s in enumerate(model.sources)}
        for stem, audio in stems.items():
            write(folder / f"{stem}.wav", audio, CLIP_RATE)
        write(folder / "nomusic.wav", stems["vocals"], CLIP_RATE)
        rows.append(result_row("demucs_baseline", clip, wav.shape[1] / model.samplerate, seconds, peak, device))
    return rows


def result_row(name, clip, length, seconds, peak, device) -> dict:
    row = {"model": name, "clip": clip.stem, "device": device.type,
           "audio_s": round(length, 1), "run_s": round(seconds, 1),
           "speed_x_realtime": round(length / seconds, 2), "peak_gpu_mb": round(peak)}
    print(row)
    return row


def main() -> None:
    wanted_clips = [a for a in sys.argv[1:] if a.startswith("clip")]
    clips = sorted(c for c in CLIPS.glob("clip*.wav") if not wanted_clips or c.stem in wanted_clips)
    if not clips:
        sys.exit("no clips: run experiments/make_clips.py first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [a for a in sys.argv[1:] if not a.startswith("clip")] or [*CANDIDATES, "demucs_baseline"]

    results_file = OUT / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else []
    for name in names:
        rows = run_demucs(clips, device) if name == "demucs_baseline" else run_msst(name, clips, device)
        done = {(r["model"], r["clip"]) for r in rows}
        results = [r for r in results if (r["model"], r["clip"]) not in done] + rows
        results_file.write_text(json.dumps(results, indent=2))
        torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
