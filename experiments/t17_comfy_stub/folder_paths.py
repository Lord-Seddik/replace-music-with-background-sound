"""Minimal stand-in for ComfyUI's folder_paths, enough for ComfyUI-HunyuanVideo-Foley (T17 experiment)."""
import os

models_dir = os.environ.get("T17_FOLEY_MODELS", r"D:\music-remover-t17\weights")
supported_pt_extensions = {".safetensors", ".pth", ".pt", ".ckpt", ".bin"}
folder_names_and_paths = {}


def get_filename_list(name):
    out = []
    for d in folder_names_and_paths.get(name, ([], None))[0]:
        if os.path.isdir(d):
            out += [f for f in os.listdir(d) if os.path.splitext(f)[1] in supported_pt_extensions]
    return sorted(out)


def get_full_path(name, filename):
    for d in folder_names_and_paths.get(name, ([], None))[0]:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None
