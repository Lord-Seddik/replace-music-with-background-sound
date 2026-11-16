"""T17: video-to-audio ambience for the three gap scenes with HunyuanVideo-Foley XL.

Runs the ComfyUI-HunyuanVideo-Foley node code (phazei, commit afd2960, cloned at
D:/music-remover-t17/src/ComfyUI-HunyuanVideo-Foley) WITHOUT ComfyUI: the few
ComfyUI functions it calls come from experiments/t17_comfy_stub/. That node adds
fp8 weight storage and "block swap" (transformer blocks kept in RAM and moved to
the GPU one at a time), which is what could make the XL model fit in 4 GB.

Usage (separate venv experiments/t17-venv, which reads the project's packages
through a .pth file and adds only what the node needs):
    experiments/t17-venv/Scripts/python.exe experiments/t17_hunyuan.py --device cuda --swap 30
    ... --text-only        (no picture: text-to-audio, to compare)
    ... --load-only --device cpu   (check that everything loads, no GPU)

Weights (D:/music-remover-t17/weights/hunyuan-foley, HF cache D:/music-remover-t17/hf):
hunyuanvideo_foley_xl_fp8_e4m3fn.safetensors, vae_128d_48k_fp16.safetensors,
synchformer_state_dict_fp16.safetensors (phazei/HunyuanVideo-Foley f94bbf3),
google/siglip2-base-patch16-512 (a89f5c5), laion/larger_clap_general (ada0c23).
Writes output/t17/hunyuan[-text]/: wav per scene, listening clip, results json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NODE_PARENT = Path(r"D:\music-remover-t17\src")
FOLEY_DIR = Path(r"D:\music-remover-t17\weights\hunyuan-foley")
os.environ.setdefault("HF_HOME", r"D:\music-remover-t17\hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "t17_comfy_stub"))
sys.path.insert(0, str(NODE_PARENT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import t17_common as common  # noqa: E402

NEGATIVE = "music, melody, singing, musical instruments, speech, talking, noisy, harsh"
FPS = 25.0
SIZE = (512, 288)  # frames handed to the node; it resizes to 512x512 (SigLIP2) and 224 crop (Synchformer)


def peak_rss_gb() -> float:
    from t17_sa3 import rss_gb
    return rss_gb()


def read_frames(clip: Path, seconds: float):
    """(T, H, W, C) float 0-1 frames at 25 fps, like a ComfyUI IMAGE batch."""
    import torch
    w, h = SIZE
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip), "-t", str(seconds),
                          "-vf", f"fps={FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(frames.copy()).float().div_(255.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--swap", type=int, default=30, help="blocks kept in RAM (XL has 36); 0 = no block swap")
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--load-only", action="store_true")
    args = ap.parse_args()
    os.environ["T17_DEVICE"] = args.device

    import torch
    if args.device == "cpu":
        torch.cuda.is_available = lambda: False
    cuda = args.device.startswith("cuda")
    import folder_paths
    folder_paths.folder_names_and_paths["foley"] = ([str(FOLEY_DIR)], folder_paths.supported_pt_extensions)
    nodes = importlib.import_module("ComfyUI-HunyuanVideo-Foley.nodes")

    t0 = time.perf_counter()
    model = nodes.HunyuanModelLoader().load_model("hunyuanvideo_foley_xl_fp8_e4m3fn.safetensors",
                                                  args.precision, "fp8_e4m3fn")
    deps = nodes.HunyuanDependenciesLoader().load_dependencies("vae_128d_48k_fp16.safetensors",
                                                               "synchformer_state_dict_fp16.safetensors")[0]
    info = {"device": args.device, "swap": args.swap, "steps": args.steps, "cfg": args.cfg,
            "text_only": args.text_only, "load_s": round(time.perf_counter() - t0, 1),
            "peak_rss_after_load_gb": round(peak_rss_gb(), 2)}
    print(info, flush=True)
    if args.load_only:
        return

    out = common.ROOT / "output" / "t17" / ("hunyuan-text" if args.text_only else "hunyuan")
    out.mkdir(parents=True, exist_ok=True)
    sampler = nodes.HunyuanFoleySampler()
    swap = None if args.swap == 0 else {"blocks_to_swap": args.swap, "prefetch_blocks": args.prefetch,
                                        "use_non_blocking": False, "block_swap_debug": False}
    runs = {}
    for name, (start, seconds, label, prompt) in common.SCENES.items():
        frames = None if args.text_only else read_frames(common.SCENE_DIR / f"{name}.mkv", seconds)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        first, _ = sampler.generate_audio(model, deps, FPS, seconds, prompt, NEGATIVE, args.cfg, args.steps,
                                          "euler", 1, args.seed, True, image=frames, block_swap_args=swap)
        took = time.perf_counter() - t
        wav, sr = first["waveform"][0].numpy(), first["sample_rate"]
        sf.write(out / f"{name}.wav", wav.T, sr, subtype="FLOAT")
        run = {"seconds": seconds, "gen_s": round(took, 1), "s_per_10s_audio": round(took / seconds * 10, 1),
               "sample_rate": sr, "peak_rss_gb": round(peak_rss_gb(), 2)}
        if cuda:
            run["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            run["gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        runs[name] = run
        print(name, run, flush=True)

    del model, deps, sampler
    if cuda:
        torch.cuda.empty_cache()
    for name in common.SCENES:
        audio, sr = sf.read(out / f"{name}.wav", dtype="float32", always_2d=True)
        audio48 = common.to_48k(audio.T, sr)
        runs[name]["music_check"] = common.music_check(audio48, args.device)
        runs[name]["listen"] = str(common.make_listen_clip(name, audio48, out).relative_to(common.ROOT))
        print(name, runs[name]["music_check"], flush=True)
    info["runs"] = runs
    (out / f"results_{args.device.replace(':', '')}_swap{args.swap}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
