"""Scenes from the video: shot cuts (ffmpeg scene score), stills, and scene
labels from an image-text model (SigLIP) comparing stills with short
descriptions such as "a classroom" or "a street at night"."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

CUT_SCORE = 0.3  # ffmpeg scene score above which a frame starts a new shot
_PTS = re.compile(r"pts_time:([0-9.]+)")

STILL_WIDTH = 320  # stills are scaled to this width; the model resizes them again

CLOSE_UP = "close_up"  # label meaning "the place is not visible in this piece"

# Scene types to tell apart: short name -> description given to the model.
# Weather ("rain", "snow") was tried and dropped: it won on blurry close-ups.
# T18b lets a sound library bring its own names.
DEFAULT_SCENES: dict[str, str] = {
    "classroom": "a classroom",
    "school_corridor": "a school corridor",
    "schoolyard": "a schoolyard",
    "city_street": "a busy city street",
    "quiet_street": "a quiet residential street",
    "night_street": "a street at night",
    "park": "a park",
    "countryside": "fields in the countryside",
    "forest": "a forest",
    "beach": "a beach by the sea",
    "living_room": "a living room at home",
    "bedroom": "a bedroom",
    "kitchen": "a kitchen",
    "hospital": "a hospital room",
    "shop": "a shop or a cafe",
    "train": "inside a train or a bus",
    "workplace": "a workplace or a construction site",
    # No place visible. Without this choice, close-ups went to whichever label
    # was vaguest ("rain" took 54 of 180 pieces in Clannad E20; see PLAN T18a).
    CLOSE_UP: "a close-up of a character's face",
}
PROMPT = "a still from an anime showing {}"

SIGLIP_REPO = "google/siglip-base-patch16-224"  # Apache-2.0
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"  # pinned 2026-11-16


def scene_cuts(path: str | Path, threshold: float = CUT_SCORE) -> list[float]:
    """Times in seconds where a new shot starts, in order.

    Frames are scaled down before scoring, which is ~17x real time for a
    1080p HEVC episode on the test laptop and does not change the cuts found.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"scale=256:-2,select='gt(scene,{threshold})',metadata=print:file=-",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed: {proc.stderr.strip()}")
    return [float(t) for t in _PTS.findall(proc.stdout)]


def still_times(start: float, end: float, count: int = 3) -> list[float]:
    """`count` evenly spread times strictly inside [start, end), away from the edges."""
    if end <= start or count < 1:
        raise ValueError("need end > start and count >= 1")
    step = (end - start) / count
    return [round(start + step * (i + 0.5), 3) for i in range(count)]


def grab_frame(path: str | Path, time: float, width: int = STILL_WIDTH) -> np.ndarray:
    """One RGB still (height, width, 3) uint8 at `time` seconds.

    Seeks before decoding, which ffmpeg does exactly (from the previous
    keyframe), so each still costs one short decode rather than a full pass.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{time:.3f}", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1",
        "-vf", f"scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not grab a frame at {time:.1f} s: "
                           f"{proc.stderr.decode('utf-8', 'replace').strip()}")
    data = proc.stdout
    if not data or len(data) % (width * 3):
        raise ValueError(f"no video frame at {time:.1f} s")
    return np.frombuffer(bytearray(data), np.uint8).reshape(-1, width, 3)


def grab_frames(path: str | Path, times: Sequence[float], width: int = STILL_WIDTH,
                workers: int = 4) -> list[np.ndarray]:
    """`grab_frame` for many times, a few ffmpeg processes at once, in order."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda t: grab_frame(path, t, width), times))


class SceneModel(Protocol):
    def scores(self, images: Sequence[np.ndarray], texts: Sequence[str]) -> np.ndarray:
        """Match logits, shape (len(images), len(texts)); higher = better match."""


@dataclass(frozen=True)
class SceneLabel:
    label: str
    score: float      # share of the match given to `label` among all labels (0-1)
    runner_up: str
    runner_up_score: float
    shares: dict[str, float] = field(default_factory=dict, compare=False)  # every label's share
    night: bool | None = None  # with `day_night`: the stills look like night rather than day


# Asked alongside the scenes when `label_scenes(day_night=True)`; compared with each
# other only (T25: right for lit night streets where brightness alone was not).
NIGHT_TEXT, DAY_TEXT = "a scene at night, in the dark", "a scene in daytime, in daylight"


def label_scenes(
    stills: Sequence[Sequence[np.ndarray]],
    model: SceneModel,
    scenes: dict[str, str] = DEFAULT_SCENES,
    prompt: str = PROMPT,
    day_night: bool = False,
) -> list[SceneLabel]:
    """One label per group of stills (a group = the stills of one gap piece).

    The model is run once over all stills; a group's logits are averaged over
    its stills, so one odd frame (a close-up, a fade) does not decide alone.
    With `day_night`, two more descriptions (night / day) are scored in the same
    call and compared only with each other, filling `SceneLabel.night`.
    """
    if len(scenes) < 2:
        raise ValueError("need at least two scene types to choose between")
    if any(len(g) == 0 for g in stills):
        raise ValueError("every group needs at least one still")
    names = list(scenes)
    flat = [img for group in stills for img in group]
    if not flat:
        return []
    texts = [prompt.format(scenes[n]) for n in names]
    if day_night:
        texts += [prompt.format(NIGHT_TEXT), prompt.format(DAY_TEXT)]
    logits = np.asarray(model.scores(flat, texts), np.float64)
    out, i = [], 0
    for group in stills:
        mean = logits[i:i + len(group)].mean(axis=0)
        i += len(group)
        night = bool(mean[len(names)] > mean[len(names) + 1]) if day_night else None
        mean = mean[:len(names)]
        share = np.exp(mean - mean.max())
        share /= share.sum()
        best, second = np.argsort(share)[::-1][:2]
        out.append(SceneLabel(names[best], round(float(share[best]), 3),
                              names[second], round(float(share[second]), 3),
                              {n: float(v) for n, v in zip(names, share)}, night))
    return out


def to_pixels(image: np.ndarray, size: int = 224):
    """SigLIP input for one RGB uint8 image: (3, size, size) float in [-1, 1]
    (plain bicubic resize, no crop, mean/std 0.5 as in the model's config)."""
    import torch
    import torch.nn.functional as F

    x = torch.tensor(np.asarray(image, np.uint8)).permute(2, 0, 1)[None].float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return ((x.clamp(0, 1) - 0.5) / 0.5)[0]


class SiglipScenes:
    """SigLIP base (~800 MB download, ~0.7 GB GPU memory in fp16, ~220 stills/s
    on the test laptop's RTX 3050; ~7 stills/s on CPU). Weights are loaded
    from safetensors only, at a pinned revision."""

    def __init__(self, device: str = "auto", batch: int = 32) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        from transformers import SiglipModel
        from transformers.utils import logging as hf_logging

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.batch = device, batch
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = Tokenizer.from_file(
            hf_hub_download(SIGLIP_REPO, "tokenizer.json", revision=SIGLIP_REVISION))
        self.tokenizer.enable_truncation(64)
        self.tokenizer.enable_padding(pad_id=1, pad_token="</s>", length=64)
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the config's unused CLIP token ids trigger a harmless warning
        hf_logging.disable_progress_bar()
        try:
            self.model = SiglipModel.from_pretrained(
                SIGLIP_REPO, revision=SIGLIP_REVISION, use_safetensors=True, dtype=self.dtype,
            ).to(device).eval()
        finally:
            hf_logging.set_verbosity(verbosity)
            hf_logging.enable_progress_bar()

    @staticmethod
    def _features(out):
        import torch
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    def scores(self, images: Sequence[np.ndarray], texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            ids = torch.tensor([e.ids for e in self.tokenizer.encode_batch(list(texts))],
                               device=self.device)
            text = self._features(self.model.get_text_features(input_ids=ids))
            rows = []
            for i in range(0, len(images), self.batch):
                pixels = torch.stack([to_pixels(im) for im in images[i:i + self.batch]])
                pixels = pixels.to(self.device, self.dtype)
                rows.append(self._features(self.model.get_image_features(pixel_values=pixels)))
            image = torch.cat(rows)
            logits = image @ text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
        return logits.cpu().numpy()
