"""T23 throwaway: label the T18a gap pieces against a real library, and make
contact sheets for checking the scenes by eye.

Usage:
    python experiments/t23_labels.py LIBRARY_DIR OUT_DIR

Reads the 180 pieces and their 3 stills from T18a (output/t18a/stills.npz,
labels.csv) and the gaps from T15 (output/t15/gaps.csv). Labels each piece
with SigLIP over the library's scene names (as `make_fill` does), resolves
close-ups with `plan_fill`, and writes OUT_DIR/labels.csv plus contact
sheets OUT_DIR/sheet_N.jpg (middle still of 30 pieces each, 6 across,
in piece order).
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from musicremover.gaps import Gap  # noqa: E402
from musicremover.library import load_library, plan_fill  # noqa: E402
from musicremover.scenes import SiglipScenes, label_scenes  # noqa: E402

PER_SHEET, ACROSS = 30, 6


def main() -> None:
    library, out = load_library(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "output/t18a/labels.csv")))
    gaps = list(csv.DictReader(open(ROOT / "output/t15/gaps.csv")))
    z = np.load(ROOT / "output/t18a/stills.npz")
    stills = z["stills"].reshape(len(rows), 3, *z["stills"].shape[1:])

    pieces = []
    for r in rows:
        s, e = float(r["start"]), float(r["end"])
        gi = next(i for i, g in enumerate(gaps) if float(g["start"]) - 0.01 <= s < float(g["end"]))
        g = gaps[gi]
        pieces.append((gi, Gap(s, e, float(g["music_db"]), float(g["effects_db"]))))

    labels = label_scenes([list(g) for g in stills], SiglipScenes(), library.scenes())
    placements, _ = plan_fill(pieces, [lb.label for lb in labels])
    resolved = []
    for _, p in pieces:
        hit = [pl.scene for pl in placements if pl.start <= p.start + 0.01 and p.end <= pl.end + 0.01]
        resolved.append(hit[0] if hit else "")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "start", "end", "gap", "label", "score", "runner_up", "runner_up_score", "fill"])
        for i, ((gi, p), lb, fill) in enumerate(zip(pieces, labels, resolved)):
            w.writerow([i, p.start, p.end, gi, lb.label, lb.score, lb.runner_up, lb.runner_up_score, fill])

    h, wd = stills.shape[2], stills.shape[3]
    for k in range(0, len(rows), PER_SHEET):
        group = stills[k:k + PER_SHEET, 1]
        if len(group) < PER_SHEET:
            group = np.concatenate([group, np.zeros((PER_SHEET - len(group), h, wd, 3), np.uint8)])
        grid = group.reshape(PER_SHEET // ACROSS, ACROSS, h, wd, 3).transpose(0, 2, 1, 3, 4)
        grid = grid.reshape(PER_SHEET // ACROSS * h, ACROSS * wd, 3)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{grid.shape[1]}x{grid.shape[0]}", "-i", "-", "-q:v", "4",
                        str(out / f"sheet_{k // PER_SHEET}.jpg")], input=np.ascontiguousarray(grid).tobytes(),
                       check=True)
    print(f"{len(rows)} pieces; sheets: {(len(rows) + PER_SHEET - 1) // PER_SHEET}")


if __name__ == "__main__":
    main()
