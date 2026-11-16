"""T23 throwaway: try fill rules offline against the eye labels (t23_truth.py).

Usage:
    python experiments/t23_sim.py LIBRARY_DIR

SigLIP logits for every T18a piece against every candidate scene are cached
in output/t23/logits.npz; then each rule picks a scene per piece, close-ups
borrow as `plan_fill` does, and the result is scored: seconds of wrong fill
and seconds of fill kept.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from musicremover.gaps import Gap  # noqa: E402
from musicremover.library import load_library, plan_fill  # noqa: E402
from musicremover.scenes import CLOSE_UP, DEFAULT_SCENES, PROMPT  # noqa: E402
from t23_truth import QUIET, TRUTH, wrong  # noqa: E402

EXTRA = {"snow": "a snowy landscape in winter", "title": "a title card with text on a plain background"}
CACHE = ROOT / "output" / "t23" / "logits.npz"


def pieces_and_stills():
    rows = list(csv.DictReader(open(ROOT / "output/t18a/labels.csv")))
    gaps = list(csv.DictReader(open(ROOT / "output/t15/gaps.csv")))
    z = np.load(ROOT / "output/t18a/stills.npz")
    stills = z["stills"].reshape(len(rows), 3, *z["stills"].shape[1:])
    pieces = []
    for r in rows:
        s, e = float(r["start"]), float(r["end"])
        gi = next(i for i, g in enumerate(gaps) if float(g["start"]) - 0.01 <= s < float(g["end"]))
        pieces.append((gi, Gap(s, e, float(gaps[gi]["music_db"]), float(gaps[gi]["effects_db"]))))
    return pieces, stills, gaps


def logits(stills, names, texts):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["names"]) == names:
            return z["logits"]
    from musicremover.scenes import SiglipScenes
    model = SiglipScenes()
    flat = [img for group in stills for img in group]
    lg = np.asarray(model.scores(flat, texts), np.float64).reshape(len(stills), 3, len(names)).mean(axis=1)
    np.savez(CACHE, logits=lg, names=np.array(names, dtype=object))
    return lg


def songs(gaps, join_s, edge_talk=0.3):
    """Song spans by the T18b rule; then each song absorbs neighbouring gaps
    closer than join_s whose own talk share is under edge_talk."""
    g = [(float(x["start"]), float(x["end"]), float(x["talking"])) for x in gaps]
    dur = 1451.0
    is_song = [(s < 300 or e > dur - 300) and e - s >= 60 and t < 0.2 for s, e, t in g]
    changed = True
    while changed and join_s > 0:
        changed = False
        for i in range(len(g)):
            if is_song[i]:
                continue
            near = (i > 0 and is_song[i - 1] and g[i][0] - g[i - 1][1] <= join_s) or \
                   (i + 1 < len(g) and is_song[i + 1] and g[i + 1][0] - g[i][1] <= join_s)
            if near and g[i][2] < edge_talk:
                is_song[i] = changed = True
    return [(s, e) for (s, e, _), f in zip(g, is_song) if f]


def run(rule, pieces, lg, names, folders, brightness, gaps, join_s):
    labels = []
    for i in range(len(pieces)):
        labels.append(rule(lg[i], names, folders, brightness[i]))
    placements, _ = plan_fill(pieces, labels)
    song = songs(gaps, join_s)
    bad = kept = good = 0.0
    for i, (_, p) in enumerate(pieces):
        hit = [pl.scene for pl in placements if pl.start <= p.start + 0.01 and p.end <= pl.end + 0.01]
        fill = hit[0] if hit else ""
        if any(a - 0.01 <= p.start < b for a, b in song):
            fill = ""
        if fill == "unsure":
            fill = "neutral"
        d = p.end - p.start
        kept += d if fill else 0
        bad += d if wrong(i, fill) else 0
        good += d if fill not in QUIET and not wrong(i, fill) else 0
    return bad, kept, good


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    library = load_library(sys.argv[1])
    folders = set(library.sounds)
    pieces, stills, gaps = pieces_and_stills()
    scenes = dict(DEFAULT_SCENES) | EXTRA
    names = list(scenes)
    lg = logits(stills, names, [PROMPT.format(scenes[n]) for n in names])
    brightness = stills.mean(axis=(1, 2, 3, 4)) / 255  # mean of the 3 stills

    def only_folders(row, names, folders, b):  # today: choose among the library's folders + close_up
        idx = [k for k, n in enumerate(names) if n in folders or n == CLOSE_UP]
        return names[idx[int(np.argmax(row[idx]))]]

    def full_set(row, names, folders, b, thr=0.0, dark=None, distinctive=frozenset()):
        p = softmax(row)
        k = int(np.argmax(p))
        n = names[k]
        if n == CLOSE_UP:
            return n
        if n not in folders:
            return "unsure"
        if n in distinctive and p[k] < thr:
            return "unsure"
        if dark is not None and b < dark and n in DAY:
            return "unsure"
        return n

    DAY = {"schoolyard", "quiet_street", "city_street", "park", "beach", "countryside", "forest"}
    DIST = DAY | {"night_street"}
    print("pieces with truth 'song' excluded from kept-time totals are counted as wrong if filled")
    print(f"{'rule':52s} {'wrong s':>8s} {'filled s':>9s} {'right scene-specific s':>9s}")
    for join_s in (0.0, 4.0):
        b0, k0, g0 = run(only_folders, pieces, lg, names, folders, brightness, gaps, join_s)
        print(f"{'today (folders only)':40s} join {join_s:3.0f}  {b0:8.0f} {k0:9.0f} {g0:9.0f}")
        for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            for dark in (None, 0.25):
                rule = lambda r, n, f, b, t=thr, d=dark: full_set(r, n, f, b, t, d, DIST)
                bad, kept, good = run(rule, pieces, lg, names, folders, brightness, gaps, join_s)
                print(f"full set, distinctive >= {thr:.2f}, dark < {str(dark):5s} join {join_s:3.0f}  {bad:8.0f} {kept:9.0f} {good:9.0f}")


if __name__ == "__main__":
    main()
