"""T25 throwaway: how well can the space (room / hall / town / nature / night)
of each gap piece be told from the cached SigLIP scene logits (t23_sim.py)?

Usage: python experiments/t25_sim.py

Eye truth comes from the T23 classes (t23_truth.py). Indoor/outdoor mix-ups
are counted apart from room/hall or town/nature ones.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from musicremover.library import plan_fill  # noqa: E402
from musicremover.scenes import CLOSE_UP  # noqa: E402
import t23_sim as t  # noqa: E402
from t23_truth import TRUTH  # noqa: E402

INDOOR, OUTDOOR = {"room", "hall"}, {"town", "nature", "night"}
TRUE_SPACE = {  # class -> acceptable spaces (first = best); None = anything goes
    "night_out": {"night"}, "home_in": {"room"}, "school_in": {"hall"}, "school_out": {"town"},
    "ball_out": {"town", "nature"}, "street_day": {"town"}, "snow": OUTDOOR, "dream": None,
    "title": None, "song": None,
}
SPACE_OF = {
    "bedroom": "room", "kitchen": "room", "living_room": "room",
    "classroom": "hall", "school_corridor": "hall", "hospital": "hall", "shop": "hall",
    "workplace": "hall", "train": "hall",
    "city_street": "town", "quiet_street": "town", "schoolyard": "town", "night_street": "night",
    "park": "nature", "countryside": "nature", "forest": "nature", "beach": "nature", "snow": "nature",
    "title": None,
}


def score(spaces, pieces):
    labels = [s if s else CLOSE_UP for s in spaces]
    placements, _ = plan_fill(pieces, labels)
    severe = minor = ok = 0.0
    for i, (_, p) in enumerate(pieces):
        want = TRUE_SPACE[TRUTH[i]]
        if want is None:
            continue
        hit = [pl.scene for pl in placements if pl.start <= p.start + 0.01 and p.end <= pl.end + 0.01]
        got = hit[0] if hit else None
        d = p.end - p.start
        if got in want:
            ok += d
        elif got is None or (got in INDOOR) != (next(iter(want)) in INDOOR):
            severe += d
        else:
            minor += d
    return ok, minor, severe


def main() -> None:
    pieces, stills, gaps = t.pieces_and_stills()
    z = np.load(t.CACHE, allow_pickle=True)
    lg, names = z["logits"], list(z["names"])
    br = stills.mean(axis=(1, 2, 3, 4)) / 255
    print(f"{'rule':40s} {'right s':>8s} {'near s':>7s} {'in/out wrong s':>15s}")
    print(f"{'one bed (room) for all':40s}", *(f"{v:7.0f}" for v in score(["room"] * len(pieces), pieces)))
    for dark in (None, 0.25, 0.3, 0.35):
        spaces = []
        for i in range(len(pieces)):
            p = t.softmax(lg[i])
            if names[int(np.argmax(p))] == CLOSE_UP:
                spaces.append(None)
                continue
            sums = {}
            for n, v in zip(names, p):
                s = SPACE_OF.get(n)
                if s:
                    sums[s] = sums.get(s, 0) + v
            s = max(sums, key=sums.get)
            if dark is not None and br[i] < dark and s in OUTDOOR:
                s = "night"
            spaces.append(s)
        print(f"{'space sums, dark < ' + str(dark):40s}", *(f"{v:7.0f}" for v in score(spaces, pieces)))


if __name__ == "__main__":
    main()
