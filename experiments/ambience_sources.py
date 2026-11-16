"""T16: how much of the episode's own ambience could fill the gaps?

Usage:
    python experiments/ambience_sources.py samples/episode.mkv

Uses the full-episode stems from the T1 scan (output/bakeoff/scan/). Finds
"source" frames (no speech, no music, some background), then counts for
each gap piece whether a source lies in the same shot or within a window.
Scene cuts are cached in output/t16/cuts.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from musicremover.gaps import _runs, find_gaps, frame_levels, split_at_cuts  # noqa: E402
from musicremover.scenes import scene_cuts  # noqa: E402

SCAN = ROOT / "output" / "bakeoff" / "scan"
OUT = ROOT / "output" / "t16"
SR = 48000


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def grow(mask: np.ndarray, frames: int) -> np.ndarray:
    """True wherever mask is True within `frames` frames."""
    if frames <= 0:
        return mask
    return np.convolve(mask.astype(float), np.ones(2 * frames + 1), mode="same") > 0


def main() -> None:
    episode = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    stems = {n: sf.read(SCAN / f"{n}.wav", dtype="float32", always_2d=True)[0].T.copy()
             for n in ("speech", "music", "effects")}
    cache = OUT / "cuts.json"
    if cache.exists():
        cuts = json.loads(cache.read_text())
    else:
        cuts = scene_cuts(episode)
        cache.write_text(json.dumps(cuts))
    lv = {n: frame_levels(a, SR) for n, a in stems.items()}
    n = len(lv["speech"])
    duration = n / 10
    pieces = split_at_cuts(find_gaps(stems, SR), cuts)
    shot_of = np.searchsorted(np.array(cuts), np.arange(n) / 10, side="right")

    print(f"{duration / 60:.1f} min, {len(cuts)} cuts, {len(pieces)} gap pieces, "
          f"{sum(p.length for p in pieces) / 60:.1f} min of gaps")
    print("effects level percentiles (dB):",
          {q: round(float(np.percentile(lv['effects'], q)), 1) for q in (10, 25, 50, 75, 90)})
    for speech_max, music_max, floor, guard_s, min_run_s in (
        (-60, -50, -70, 0.5, 1.0),
        (-55, -50, -70, 0.5, 1.0),
        (-60, -55, -65, 0.5, 1.0),
        (-60, -50, -75, 0.3, 0.5),
        (-50, -45, -75, 0.3, 0.5),
    ):
        g = round(guard_s * 10)
        bad = grow(lv["speech"] > speech_max, g) | grow(lv["music"] > music_max, g)
        mask = ~bad & (lv["effects"] > floor)
        runs = [(a, b) for a, b in _runs(mask) if (b - a) / 10 >= min_run_s]
        src = np.zeros(n, bool)
        for a, b in runs:
            src[a:b] = True
        total = src.sum() / 10
        shots_with = set(shot_of[src])
        same_shot = sum(1 for p in pieces if shot_of[int(p.start * 10)] in shots_with)
        idx = np.flatnonzero(src) / 10

        def near(p, w):
            return len(idx) and np.any((idx > p.start - w) & (idx < p.end + w))

        w30 = [p for p in pieces if near(p, 30)]
        w60 = [p for p in pieces if near(p, 60)]
        print(f"speech<{speech_max} music<{music_max} effects>{floor} guard {guard_s}s run>={min_run_s}s: "
              f"source {total:.0f} s in {len(runs)} runs, {len(shots_with)} shots | pieces with source: "
              f"same shot {same_shot} ({sum(p.length for p in pieces if shot_of[int(p.start * 10)] in shots_with):.0f} s), "
              f"<=30 s {len(w30)} ({sum(p.length for p in w30):.0f} s), <=60 s {len(w60)} ({sum(p.length for p in w60):.0f} s)")
        if speech_max == -60 and music_max == -50 and floor == -70:
            for a, b in runs:
                lev = 10 * np.log10(np.mean(10 ** (lv["effects"][a:b] / 10)))
                print(f"    {clock(a / 10):>7} - {clock(b / 10):>7}  {(b - a) / 10:4.1f} s  effects {lev:6.1f} dB")


if __name__ == "__main__":
    main()
