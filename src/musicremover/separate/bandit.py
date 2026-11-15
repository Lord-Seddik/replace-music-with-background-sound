"""Bandit v2 backend: the cinematic speech / music / effects model chosen in T1.

Weights: `checkpoint-multi.ckpt` (multilingual) from WEIGHTS_URL, licensed
CC-BY-SA 4.0, so they are downloaded by the user rather than shipped here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

WEIGHTS_URL = "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt"
SAMPLE_RATE = 48000
CHUNK_SIZE = 384000  # 8 s, what the model was trained on

# From configs/config_dnr_bandit_v2_mus64.yaml in ZFTurbo/Music-Source-Separation-Training.
MODEL_KWARGS = dict(
    in_channels=1, stems=["speech", "music", "sfx"], band_type="musical", n_bands=64,
    normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=8,
    emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="GRU", mlp_dim=512,
    hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True,
    use_freq_weights=True, n_fft=2048, win_length=2048, hop_length=512,
    window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True,
    pad_mode="reflect", onesided=True, fs=SAMPLE_RATE,
)
MODEL_STEMS = {"speech": 0, "music": 1, "effects": 2}  # STEMS name -> model output index


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint's tensors. weights_only refuses files that carry code."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "state", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    return state


def match_keys(state: dict[str, torch.Tensor], wanted: set[str]) -> dict[str, torch.Tensor]:
    """Strip whatever prefix the training framework added to parameter names."""
    for prefix in ("", "model.", "_orig_mod.", "model._orig_mod."):
        trial = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if wanted <= set(trial):
            return {k: trial[k] for k in wanted}
    missing = sorted(wanted - set(state))[:5]
    raise RuntimeError(f"checkpoint does not match Bandit v2, e.g. missing {missing}")


class BanditV2:
    sample_rate = SAMPLE_RATE
    chunk_size = CHUNK_SIZE

    def __init__(self, weights: Path, device: str = "auto") -> None:
        from .bandit_v2 import Bandit

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"model weights not found at {weights}. Download checkpoint-multi.ckpt from {WEIGHTS_URL}"
            )
        self.device = pick_device(device)
        model = Bandit(**MODEL_KWARGS)
        model.load_state_dict(match_keys(load_state(weights), set(model.state_dict())))
        self.model = model.to(self.device).eval()
        self.order = [MODEL_STEMS[name] for name in ("speech", "music", "effects")]

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None].to(self.device)
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            y = self.model(x)  # (1, stems, channels, samples)
        return y[0, self.order].float().cpu().numpy()
