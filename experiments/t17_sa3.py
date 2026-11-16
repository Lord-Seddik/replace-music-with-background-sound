"""T17: generate ambience for the three gap scenes with Stable Audio 3 Small SFX.

Text-to-audio: the prompt comes from the scene's SigLIP label (T18a), not
from the picture itself.

Usage (project Python, nothing installed into .venv):
    .venv/Scripts/python.exe experiments/t17_sa3.py --device cpu   [--load-only]
    .venv/Scripts/python.exe experiments/t17_sa3.py --device cuda  [--half]

Code: Stability-AI/stable-audio-3 cloned at D:/music-remover-t17/src/stable-audio-3
(commit 790793b), put on sys.path. Weights: D:/music-remover-t17/weights/sa3-small-sfx
(sha256 checked against stabilityai/stable-audio-3-small-sfx). Writes output/t17/sa3/:
raw wav per scene, a listening clip (no-music track + ambience at the T18b level),
a 10 s timing run, and results.json (time, memory, music check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SA3_SRC = Path(r"D:\music-remover-t17\src\stable-audio-3")
WEIGHTS = Path(r"D:\music-remover-t17\weights\sa3-small-sfx")
sys.path.insert(0, str(SA3_SRC))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import t17_common as common  # noqa: E402

OUT = common.ROOT / "output" / "t17" / "sa3"


def rss_gb() -> float:
    """Peak working set of this process (Windows), in GB."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        pmc = PMC(cb=ctypes.sizeof(PMC))
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        get = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        get(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1e9
    except (OSError, AttributeError):
        return float("nan")


def load(device: str, half: bool):
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel
    config = json.loads((WEIGHTS / "model_config.json").read_text())
    for c in config["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":  # read the text encoder from disk, never from the Hub
            c["config"].pop("repo_id", None)
            c["config"].pop("subfolder", None)
            c["config"]["model_path"] = str(WEIGHTS / "t5gemma-b-b-ul2")
    model = load_diffusion_cond(config, str(WEIGHTS / "model.safetensors"), device=device, model_half=half)
    model.use_lora, model.lora_names = False, []
    return StableAudioModel(model, config, device, half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half", action="store_true", help="fp16 weights (GPU only)")
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--music-check-device", default=None, help="device for Bandit (default: same)")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.cuda.is_available = lambda: False  # keep everything off the GPU
    cuda = args.device.startswith("cuda")

    t0 = time.perf_counter()
    model = load(args.device, args.half and cuda)
    load_s = time.perf_counter() - t0
    info = {"device": args.device, "half": bool(args.half and cuda), "load_s": round(load_s, 1),
            "peak_rss_after_load_gb": round(rss_gb(), 2)}
    if cuda:
        info["gpu_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    print(info, flush=True)
    if args.load_only:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    jobs = [(n, s[1], s[3]) for n, s in common.SCENES.items()]
    jobs.append(("timing_10s_" + jobs[0][0], 10.0, jobs[0][2]))
    for name, seconds, prompt in jobs:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t = time.perf_counter()
        audio = model.generate(prompt=prompt, duration=seconds, seed=args.seed)
        if cuda:
            torch.cuda.synchronize()
        took = time.perf_counter() - t
        audio = audio[0].cpu().numpy()  # (channels, samples)
        sr = model.model.sample_rate
        sf.write(OUT / f"{name}.wav", audio.T, sr, subtype="FLOAT")
        run = {"seconds": seconds, "gen_s": round(took, 2), "s_per_10s_audio": round(took / seconds * 10, 2),
               "sample_rate": sr, "peak_rss_gb": round(rss_gb(), 2)}
        if cuda:
            run["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            run["gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        runs[name] = run
        print(name, run, flush=True)

    del model
    if cuda:
        torch.cuda.empty_cache()
    for name in common.SCENES:
        audio, sr = sf.read(OUT / f"{name}.wav", dtype="float32", always_2d=True)
        audio48 = common.to_48k(audio.T, sr)
        runs[name]["music_check"] = common.music_check(audio48, args.music_check_device or args.device)
        runs[name]["listen"] = str(common.make_listen_clip(name, audio48, OUT).relative_to(common.ROOT))
        print(name, runs[name]["music_check"], flush=True)
    info["runs"] = runs
    (OUT / f"results_{args.device.replace(':', '')}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
