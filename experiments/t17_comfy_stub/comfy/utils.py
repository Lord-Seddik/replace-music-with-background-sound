import torch


def load_torch_file(path, safe_load=True, device=None):
    """safetensors via safetensors, anything else with weights_only=True (no pickled code)."""
    device = device or torch.device("cpu")
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device=str(device))
    sd = torch.load(path, map_location=device, weights_only=True)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


class ProgressBar:
    def __init__(self, total, *args, **kwargs):
        self.total, self.current = total, 0

    def update(self, n=1):
        self.current += n

    def update_absolute(self, value, total=None, preview=None):
        self.current = value
