"""T25: score the real fill code on the eye labels: wrong scene sound (T23 rule)
and, for the neutral beds, wrong space (indoor/outdoor mix-ups apart).

Usage: python experiments/t25_check.py LIBRARY_DIR
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from musicremover.library import NEUTRAL, choose_scene, load_library, neutral_bed, plan_fill, space_of  # noqa: E402
from musicremover.scenes import SiglipScenes, label_scenes  # noqa: E402
import t23_sim as t  # noqa: E402
from t23_truth import QUIET, TRUTH, wrong  # noqa: E402
from t25_sim import INDOOR, TRUE_SPACE  # noqa: E402

lib = load_library(sys.argv[1])
pieces, stills, gaps = t.pieces_and_stills()
labels = label_scenes([list(g) for g in stills], SiglipScenes(), lib.scenes(), day_night=True)
chosen = []
for lb, g in zip(labels, stills):
    b = float(np.mean([s.mean() for s in g])) / 255
    sc = choose_scene(lb, b, lib)
    chosen.append(neutral_bed(space_of(lb, b), lib) if sc == NEUTRAL else sc)
pl, _ = plan_fill(pieces, chosen)
song = t.songs(gaps, 4)
bad = kept = neu = sp_ok = sp_near = sp_bad = 0.0
for i, (_, p) in enumerate(pieces):
    hit = [x.scene for x in pl if x.start <= p.start + .01 and p.end <= x.end + .01]
    f = hit[0] if hit else ""
    if any(a - .01 <= p.start < b for a, b in song):
        f = ""
    d = p.end - p.start
    kept += d if f else 0
    is_neutral = f.startswith(NEUTRAL)
    bad += d if wrong(i, NEUTRAL if is_neutral else f) else 0
    if is_neutral:
        neu += d
        want, got = TRUE_SPACE[TRUTH[i]], f.split("_", 1)[1] if "_" in f else "room"
        if want is None or got in want:
            sp_ok += d
        elif (got in INDOOR) == (next(iter(want)) in INDOOR):
            sp_near += d
        else:
            sp_bad += d
print(f"wrong scene sound {bad:.0f} of {kept:.0f} s filled; neutral beds {neu:.0f} s: right space {sp_ok:.0f}, "
      f"near (room/hall, town/nature/night) {sp_near:.0f}, indoor/outdoor wrong {sp_bad:.0f}")
