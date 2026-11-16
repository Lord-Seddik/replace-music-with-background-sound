import os

import torch


def get_torch_device():
    return torch.device(os.environ.get("T17_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def unet_offload_device():
    return torch.device("cpu")


def soft_empty_cache(force=False):
    if torch.cuda.is_available() and os.environ.get("T17_DEVICE", "cuda").startswith("cuda"):
        torch.cuda.empty_cache()
