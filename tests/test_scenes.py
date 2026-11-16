"""Stills and scene labels. A fake model stands in for SigLIP, so nothing is
downloaded; only `to_pixels` needs torch."""
import subprocess

import numpy as np
import pytest

from musicremover.scenes import (
    CLOSE_UP, DEFAULT_SCENES, PROMPT, grab_frame, label_scenes, still_times, to_pixels,
)


def test_still_times_spread_inside_the_piece():
    assert still_times(10.0, 16.0, 3) == [11.0, 13.0, 15.0]
    assert still_times(0.0, 0.5, 1) == [0.25]
    with pytest.raises(ValueError):
        still_times(5.0, 5.0)


@pytest.fixture(scope="module")
def red_then_blue(tmp_path_factory):
    """2 s red, then 2 s blue, 160x120."""
    out = tmp_path_factory.mktemp("stills") / "rb.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=red:size=160x120:rate=10:duration=2",
        "-f", "lavfi", "-i", "color=c=blue:size=160x120:rate=10:duration=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
        "-c:v", "libx264", "-g", "40", "-pix_fmt", "yuv420p", str(out),
    ], check=True)
    return out


def test_grab_frame_seeks_to_the_time(red_then_blue):
    red = grab_frame(red_then_blue, 1.0, width=64)
    blue = grab_frame(red_then_blue, 3.0, width=64)  # past the only keyframe: exact seek needed
    assert red.shape == (48, 64, 3) and red.dtype == np.uint8
    assert red[..., 0].mean() > 200 and red[..., 2].mean() < 60
    assert blue[..., 2].mean() > 200 and blue[..., 0].mean() < 60
    red[0, 0] = 0  # writable


def test_grab_frame_past_the_end_fails(red_then_blue):
    with pytest.raises(ValueError):
        grab_frame(red_then_blue, 30.0)


class ColourModel:
    """Scores 'red' texts by redness and 'blue' texts by blueness."""

    def __init__(self):
        self.calls = []

    def scores(self, images, texts):
        self.calls.append((len(images), list(texts)))
        out = np.zeros((len(images), len(texts)))
        for i, im in enumerate(images):
            r, b = im[..., 0].mean() / 255, im[..., 2].mean() / 255
            for j, t in enumerate(texts):
                out[i, j] = 10 * r if "red" in t else 10 * b if "blue" in t else 0
        return out


def still(rgb):
    return np.full((8, 8, 3), rgb, np.uint8)


SCENES = {"fire": "a red room", "sea": "a blue sea", "fog": "a grey fog"}


def test_label_scenes_picks_best_match_per_group():
    model = ColourModel()
    groups = [[still((255, 0, 0))] * 3, [still((0, 0, 255)), still((0, 0, 200))]]
    labels = label_scenes(groups, model, SCENES)
    assert [lb.label for lb in labels] == ["fire", "sea"]
    assert labels[0].runner_up in ("sea", "fog")
    assert 0.9 < labels[0].score <= 1 and labels[0].runner_up_score < 0.1
    assert model.calls == [(5, [PROMPT.format(d) for d in SCENES.values()])]  # one model call


def test_label_scenes_averages_over_the_stills():
    # one blue frame among three red ones does not flip the label
    groups = [[still((255, 0, 0)), still((255, 0, 0)), still((0, 0, 255))]]
    (lb,) = label_scenes(groups, ColourModel(), SCENES)
    assert lb.label == "fire" and lb.runner_up == "sea"
    assert 0.5 < lb.score < 1


def test_label_scenes_rejects_bad_input():
    with pytest.raises(ValueError):
        label_scenes([[still((1, 2, 3))]], ColourModel(), {"only": "one"})
    with pytest.raises(ValueError):
        label_scenes([[]], ColourModel(), SCENES)
    assert label_scenes([], ColourModel(), SCENES) == []


def test_default_scenes_are_distinct_short_names():
    assert len(DEFAULT_SCENES) >= 10 and CLOSE_UP in DEFAULT_SCENES
    assert len(set(DEFAULT_SCENES.values())) == len(DEFAULT_SCENES)
    assert all(n.isidentifier() and n.islower() for n in DEFAULT_SCENES)


def test_to_pixels_matches_siglip_input():
    torch = pytest.importorskip("torch")
    img = np.zeros((90, 160, 3), np.uint8)
    img[..., 0] = 255
    x = to_pixels(img)
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert torch.allclose(x[0], torch.ones(224, 224)) and torch.allclose(x[1], -torch.ones(224, 224))
