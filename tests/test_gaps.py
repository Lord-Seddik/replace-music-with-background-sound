import subprocess

import numpy as np
import pytest

from musicremover.gaps import Gap, find_gaps, frame_levels, split_at_cuts
from musicremover.scenes import scene_cuts

RATE = 8000  # small rate keeps tests fast; the functions take any rate


def level(db: float, seconds: float) -> np.ndarray:
    """Stereo noise at a given RMS level in dB."""
    n = int(seconds * RATE)
    x = np.random.default_rng(int(seconds * 10)).standard_normal((2, n)).astype(np.float32)
    return x * np.float32(10 ** (db / 20))


def silence(seconds: float) -> np.ndarray:
    return np.zeros((2, int(seconds * RATE)), np.float32)


def stems(music, effects):
    return {"speech": np.zeros_like(music), "music": music, "effects": effects}


def test_frame_levels_measures_db():
    lv = frame_levels(level(-20, 1), RATE)
    assert len(lv) == 10
    assert np.allclose(lv, -20, atol=1.0)


def test_music_with_no_background_is_a_gap():
    music = np.concatenate([silence(3), level(-30, 6), silence(3)], axis=1)
    gaps = find_gaps(stems(music, silence(12)), RATE)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.start == pytest.approx(3, abs=0.6)
    assert g.end == pytest.approx(9, abs=0.6)
    assert g.music_db == pytest.approx(-31, abs=2)


def test_music_under_loud_background_is_not_a_gap():
    music = level(-40, 6)
    effects = level(-35, 6)
    assert find_gaps(stems(music, effects), RATE) == []


def test_inaudible_music_is_not_a_gap():
    assert find_gaps(stems(level(-60, 6), silence(6)), RATE) == []


def test_short_gaps_dropped_and_short_holes_bridged():
    short = np.concatenate([level(-30, 1), silence(5)], axis=1)
    assert find_gaps(stems(short, silence(6)), RATE) == []
    holed = np.concatenate([level(-30, 3), silence(0.4), level(-30, 3)], axis=1)
    gaps = find_gaps(stems(holed, silence(6.4)), RATE)
    assert len(gaps) == 1 and gaps[0].length > 5.5


def test_split_at_cuts():
    gaps = [Gap(10, 20, -30, -80), Gap(30, 32, -30, -80)]
    out = split_at_cuts(gaps, [12, 19.8, 25, 31])
    assert [(g.start, g.end) for g in out] == [(10, 12), (12, 19.8), (30, 31), (31, 32)]
    # the 0.2 s piece between 19.8 and 20 is dropped


def test_scene_cuts_finds_a_hard_cut(tmp_path):
    clip = tmp_path / "cut.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin",
         "-f", "lavfi", "-i", "color=c=black:size=160x120:rate=10:duration=2",
         "-f", "lavfi", "-i", "color=c=white:size=160x120:rate=10:duration=2",
         "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-c:v", "libx264", str(clip)],
        check=True)
    cuts = scene_cuts(clip)
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(2.0, abs=0.15)


def test_scene_cuts_rejects_bad_threshold(tmp_path):
    with pytest.raises(ValueError):
        scene_cuts(tmp_path / "x.mkv", threshold=2)
