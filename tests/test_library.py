"""Ambience fill from a sound library: a tiny generated library and a fake
scene model stand in for real recordings and SigLIP."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

from musicremover.gaps import Gap, frame_levels
from musicremover.library import (
    DARK, LABEL_ONLY, LEVEL_RANGE_DB, MIN_SCORE, NEUTRAL, Library, Placement, choose_scene, excerpt,
    fill_level, load_library, make_fill, plan_fill, read_recording, render_fill, split_songs,
    subtract_ranges,
)
from musicremover.scenes import CLOSE_UP, DEFAULT_SCENES, SceneLabel

RATE = 8000


def noise(seconds, db=-20.0, channels=2, seed=0, rate=RATE):
    x = np.random.default_rng(seed).standard_normal((channels, int(seconds * rate))).astype(np.float32)
    return x * np.float32(10 ** (db / 20))


def db(x):
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12))


@pytest.fixture
def lib_dir(tmp_path):
    for name, seconds in (("park", 3.0), ("beach", 0.5 + 1.0), ("temple_yard", 2.0)):
        d = tmp_path / "lib" / name
        d.mkdir(parents=True)
        sf.write(d / "a.wav", noise(seconds, channels=1).T, RATE)
    (tmp_path / "lib" / "park" / "notes.txt").write_text("not audio")
    (tmp_path / "lib" / "empty").mkdir()
    return tmp_path / "lib"


def test_load_library_reads_scene_folders(lib_dir):
    lib = load_library(lib_dir)
    assert sorted(lib.sounds) == ["beach", "park", "temple_yard"]  # empty folder ignored
    assert [f.name for f in lib.sounds["park"]] == ["a.wav"]
    scenes = lib.scenes()
    assert scenes["park"] == DEFAULT_SCENES["park"]
    assert scenes["temple_yard"] == "a temple yard"
    assert scenes[CLOSE_UP] == DEFAULT_SCENES[CLOSE_UP]


def test_load_library_rejects_bad_folders(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_library(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no recordings"):
        load_library(tmp_path / "empty")
    d = tmp_path / "lib" / CLOSE_UP
    d.mkdir(parents=True)
    sf.write(d / "x.wav", noise(2, channels=1).T, RATE)
    with pytest.raises(ValueError, match="reserved"):
        load_library(tmp_path / "lib")


def test_split_songs_skips_long_quiet_gaps_near_the_ends():
    duration = 1500.0
    speech = np.full(int(duration * 10), -90.0)
    speech[6000:7000] = -20  # talking from 600 s to 700 s
    opening = Gap(130, 200, -30, -80)       # 70 s, no talk, near the start: song
    talky = Gap(600, 680, -30, -80)          # long but talking: kept (and not near an end)
    middle = Gap(800, 900, -30, -80)         # long, quiet, mid-episode: kept
    short_end = Gap(1380, 1410, -30, -80)    # near the end but short, and 10 s from the song: kept
    ending = Gap(1420, 1500, -30, -80)
    keep, songs = split_songs([opening, talky, middle, short_end, ending], speech, duration)
    assert songs == [opening, ending]
    assert keep == [talky, middle, short_end]


def test_split_songs_takes_in_quiet_credits_right_next_to_a_song():
    duration = 1500.0
    speech = np.full(int(duration * 10), -90.0)
    speech[11300:11400] = -20  # 1130-1140 s: talking
    song = Gap(130, 200, -30, -80)
    credits_before = Gap(110, 127, -30, -80)   # 3 s before the song, quiet: song too
    credits_before2 = Gap(95, 108, -30, -80)   # 2 s before that one: song too (chains)
    talky_next = Gap(1128, 1145, -30, -80)     # next to nothing: kept
    after_far = Gap(205, 215, -30, -80)        # 5 s after the song: kept
    end_song = Gap(1350, 1420, -30, -80)
    talky_after = Gap(1422, 1440, -30, -80)
    speech[14220:14400] = -20                  # the preview after the ending talks: kept
    keep, songs = split_songs([credits_before2, credits_before, song, after_far, talky_next, end_song,
                               talky_after], speech, duration)
    assert songs == [credits_before2, credits_before, song, end_song]
    assert keep == [after_far, talky_next, talky_after]


def label(name, score=0.9):
    return SceneLabel(name, score, "x", 0.0)


def test_library_scenes_offer_every_known_scene_but_not_neutral(lib_dir):
    (lib_dir / NEUTRAL).mkdir()
    sf.write(lib_dir / NEUTRAL / "n.wav", noise(2, channels=1).T, RATE)
    scenes = load_library(lib_dir).scenes()
    assert scenes["classroom"] == DEFAULT_SCENES["classroom"]  # no folder, still a choice
    assert set(LABEL_ONLY) <= set(scenes)
    assert NEUTRAL not in scenes


def test_choose_scene_falls_back_to_neutral_when_unsure(lib_dir):
    lib = load_library(lib_dir)  # park, beach, temple_yard
    assert choose_scene(label("park"), 0.6, lib) == "park"
    assert choose_scene(label(CLOSE_UP, 0.2), 0.6, lib) == CLOSE_UP
    assert choose_scene(label("classroom"), 0.6, lib) == NEUTRAL       # no folder
    assert choose_scene(label("snow"), 0.6, lib) == NEUTRAL            # label-only choice
    assert choose_scene(label("park", MIN_SCORE - 0.01), 0.6, lib) == NEUTRAL  # unsure, distinctive
    assert choose_scene(label("temple_yard", MIN_SCORE - 0.01), 0.6, lib) == NEUTRAL  # unknown kind
    assert choose_scene(label("park"), DARK - 0.01, lib) == NEUTRAL    # day sound on a night shot
    assert choose_scene(label("temple_yard"), DARK - 0.01, lib) == "temple_yard"


def test_choose_scene_keeps_quiet_rooms_even_when_unsure(tmp_path):
    d = tmp_path / "lib" / "bedroom"
    d.mkdir(parents=True)
    sf.write(d / "a.wav", noise(2, channels=1).T, RATE)
    lib = load_library(tmp_path / "lib")
    assert choose_scene(label("bedroom", 0.2), 0.1, lib) == "bedroom"


def test_render_fill_steady_avoids_a_loud_event(tmp_path):
    rec = noise(20, db=-40, channels=1)
    rec[:, 2 * RATE:18 * RATE] *= 0          # silence, then
    rec[:, 2 * RATE:18 * RATE] += noise(16, db=-40, channels=1, seed=5)
    rec[:, 9 * RATE:10 * RATE] *= 100        # one loud second in the middle (+40 dB)
    d = tmp_path / "lib" / "park"
    d.mkdir(parents=True)
    sf.write(d / "a.wav", rec.T, RATE)
    lib = load_library(tmp_path / "lib")
    pl = [Placement(0.0, 4.0, "park", -45.0, 0)]
    peaks = []
    for seed in range(6):
        out = render_fill(pl, lib, 1, 4 * RATE, RATE, seed=seed, steady=True)
        levels = frame_levels(out[:, RATE:3 * RATE], RATE)  # away from the fades
        peaks.append(levels.max() - np.median(levels))
    assert max(peaks) < 10  # never the loud second


def test_make_fill_leaves_unsure_pieces_empty_without_a_neutral_folder(red_blue_video, lib_dir):
    stems = {"speech": np.zeros((2, 6 * RATE), np.float32),
             "music": noise(6, db=-30), "effects": noise(6, db=-90, seed=3)}

    class Unsure:  # every piece looks like a classroom, which has no folder
        def scores(self, images, texts):
            return np.array([[5.0 if "classroom" in t else 0.0 for t in texts] for _ in images])

    fill, report = make_fill(stems, RATE, red_blue_video, load_library(lib_dir), Unsure(), cuts=[3.0])
    assert fill is None and report.placements == [] and len(report.unfilled) == 1

    (lib_dir / NEUTRAL).mkdir()
    sf.write(lib_dir / NEUTRAL / "n.wav", noise(2, channels=1).T, RATE)
    fill, report = make_fill(stems, RATE, red_blue_video, load_library(lib_dir), Unsure(), cuts=[3.0])
    assert [(p.start, p.end, p.scene) for p in report.placements] == [(0.0, 6.0, NEUTRAL)]


def test_subtract_ranges_cuts_keep_ranges_out():
    g = Gap(10, 30, -30, -80)
    assert subtract_ranges([g], [(15, 20)]) == [Gap(10, 15, -30, -80), Gap(20, 30, -30, -80)]
    assert subtract_ranges([g], [(0, 29)]) == []          # 1 s left over: too short
    assert subtract_ranges([g], [(0, float("inf"))]) == []
    assert subtract_ranges([g], [(40, 50)]) == [g]


def test_fill_level_follows_the_music_within_range():
    lo, hi = LEVEL_RANGE_DB
    assert fill_level(-38) == -50
    assert fill_level(-10) == hi
    assert fill_level(-80) == lo


def piece(a, b, music=-38.0):
    return Gap(a, b, music, -80.0)


def test_plan_fill_close_up_borrows_from_its_own_gap_first():
    pieces = [(0, piece(0, 4)), (0, piece(4, 6)), (1, piece(6.5, 8))]
    plan, unfilled = plan_fill(pieces, ["park", CLOSE_UP, "beach"])
    # the close-up is nearer to gap 1's beach piece, but takes park from its own gap
    assert [(p.start, p.end, p.scene) for p in plan] == [(0, 6, "park"), (6.5, 8, "beach")]
    assert unfilled == []


def test_plan_fill_close_up_gap_borrows_from_a_near_gap_or_stays_empty():
    pieces = [(0, piece(0, 4)), (1, piece(20, 25)), (2, piece(100, 104))]
    plan, unfilled = plan_fill(pieces, ["park", CLOSE_UP, CLOSE_UP])
    assert [(p.start, p.end, p.scene, p.gap) for p in plan] == [(0, 4, "park", 0), (20, 25, "park", 1)]
    assert unfilled == [piece(100, 104)]


def test_plan_fill_scene_change_touches_and_closes_small_holes():
    pieces = [(0, piece(0, 3)), (0, piece(3.4, 6)), (0, piece(6, 9))]
    plan, _ = plan_fill(pieces, ["park", "beach", "beach"])
    assert [(p.start, p.end, p.scene) for p in plan] == [(0, 3, "park"), (3, 9, "beach")]
    assert plan[0].level_db == fill_level(-38)


def test_plan_fill_keeps_every_real_label():
    # a short piece between two long ones of another scene keeps its own label
    pieces = [(0, piece(0, 5)), (0, piece(5, 6)), (0, piece(6, 11))]
    plan, _ = plan_fill(pieces, ["park", "beach", "park"])
    assert [p.scene for p in plan] == ["park", "beach", "park"]


def test_excerpt_long_and_looped():
    rng = np.random.default_rng(1)
    src = noise(3)
    cut = excerpt(src, RATE, rng, loop_fade=RATE // 4)
    assert cut.shape == (2, RATE)
    looped = excerpt(noise(1), 5 * RATE, rng, loop_fade=RATE // 4)
    assert looped.shape == (2, 5 * RATE)
    # no holes and no bumps at the loop joins (0.1 s windows within +-3 dB)
    levels = frame_levels(looped, RATE)
    assert levels.min() > -23 and levels.max() < -17


def fake_read(path, rate, channels):
    return noise(2, db=-20, channels=channels, seed=len(str(path))), -20.0


def test_render_fill_levels_fades_and_crossfades(tmp_path):
    lib = Library(tmp_path, {"park": (tmp_path / "p.wav",), "beach": (tmp_path / "beach.wav",)})
    plan = [Placement(2.0, 6.0, "park", -45.0, 0), Placement(6.0, 9.0, "beach", -45.0, 0)]
    out = render_fill(plan, lib, 2, 12 * RATE, RATE, read=fake_read)
    assert out.shape == (2, 12 * RATE)
    assert not out[:, :2 * RATE].any() and not out[:, 9 * RATE:].any()  # only inside the gap
    assert abs(db(out[:, 4 * RATE:5 * RATE]) - -45) < 1
    assert abs(out[:, 2 * RATE]).max() < 1e-3                          # faded in at the gap start
    levels = frame_levels(out[:, int(3.5 * RATE):int(7.5 * RATE)], RATE)
    assert levels.min() > -48.5 and levels.max() < -41.5                # crossfade: no dip, no bump


def test_render_fill_level_offset_moves_the_fill(tmp_path):
    lib = Library(tmp_path, {"park": (tmp_path / "p.wav",)})
    plan = [Placement(2.0, 6.0, "park", -45.0, 0)]
    out = render_fill(plan, lib, 2, 8 * RATE, RATE, read=fake_read, level_offset_db=6.0)
    assert abs(db(out[:, 3 * RATE:5 * RATE]) - -39) < 1


def test_render_fill_is_repeatable(tmp_path):
    lib = Library(tmp_path, {"park": (tmp_path / "a.wav", tmp_path / "bb.wav")})
    plan = [Placement(1.0, 3.0, "park", -45.0, 0)]
    a = render_fill(plan, lib, 2, 4 * RATE, RATE, read=fake_read)
    b = render_fill(plan, lib, 2, 4 * RATE, RATE, read=fake_read)
    assert np.array_equal(a, b)


def test_read_recording_resamples_and_matches_channels(tmp_path):
    f = tmp_path / "mono.wav"
    sf.write(f, np.concatenate([noise(2, -30, 1, rate=22050), np.zeros((1, 22050), np.float32)], 1).T, 22050)
    audio, level = read_recording(f, 48000, 2)
    assert audio.shape == (2, 3 * 48000) and audio.dtype == np.float32
    assert abs(level - -30) < 1  # the silent second does not lower the level
    sf.write(tmp_path / "short.wav", noise(0.5, channels=1).T, RATE)
    with pytest.raises(ValueError, match="shorter"):
        read_recording(tmp_path / "short.wav", RATE, 2)


@pytest.fixture(scope="module")
def red_blue_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("fillvid") / "rb.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=red:size=64x48:rate=10:duration=3",
        "-f", "lavfi", "-i", "color=c=blue:size=64x48:rate=10:duration=3",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ], check=True)
    return out


class ColourModel:
    """park for red stills, beach for blue ones, close-up never."""

    def scores(self, images, texts):
        out = np.zeros((len(images), len(texts)))
        for i, im in enumerate(images):
            red = im[..., 0].mean() > im[..., 2].mean()
            for j, t in enumerate(texts):
                out[i, j] = 5.0 if ("park" in t) == red and ("sea" in t) != red else 0.0
        return out


def test_make_fill_end_to_end(red_blue_video, lib_dir):
    stems = {"speech": np.zeros((2, 6 * RATE), np.float32),
             "music": noise(6, db=-30), "effects": noise(6, db=-90, seed=3)}
    built = []

    def factory():
        built.append(1)
        return ColourModel()

    fill, report = make_fill(stems, RATE, red_blue_video, load_library(lib_dir), factory, cuts=[3.0])
    assert built == [1]
    assert [(p.start, p.end, p.scene) for p in report.placements] == [(0.0, 3.0, "park"), (3.0, 6.0, "beach")]
    assert fill.shape == stems["music"].shape
    assert report.placements[0].level_db == fill_level(-30.2)
    assert abs(db(fill[:, RATE:2 * RATE]) - report.placements[0].level_db) < 1.5


def test_make_fill_nothing_to_fill_loads_no_model(red_blue_video, lib_dir):
    stems = {"speech": np.zeros((2, 6 * RATE), np.float32),
             "music": noise(6, db=-30), "effects": noise(6, db=-30, seed=3)}  # music not dominant
    fill, report = make_fill(stems, RATE, red_blue_video, load_library(lib_dir),
                             lambda: pytest.fail("model loaded"), cuts=[3.0])
    assert fill is None and report.placements == []
