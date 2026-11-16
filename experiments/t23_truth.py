"""T23: scene classes for the 180 T18a gap pieces of Clannad E20, judged by eye
from output/t23/sheet_*.jpg (Claude, 2026-11-16), and which library folders
are acceptable for each class. Coarse on purpose: the question is "would this
fill sound wrong here", not the exact room.

Usage: python experiments/t23_truth.py LABELS_CSV   (prints wrong fills)
"""
import csv
import sys

# index ranges (inclusive) -> class
_RANGES = [
    (0, 22, "night_out"), (23, 37, "song"), (38, 44, "dream"), (45, 46, "street_day"),
    (47, 47, "title"), (48, 53, "home_in"), (54, 54, "school_out"), (55, 67, "school_in"),
    (68, 68, "school_out"), (69, 79, "school_in"), (80, 86, "home_in"), (87, 87, "night_out"),
    (88, 91, "home_in"), (92, 99, "night_out"), (100, 100, "dream"), (101, 102, "snow"),
    (103, 106, "home_in"), (107, 111, "snow"), (112, 113, "home_in"), (114, 116, "night_out"),
    (117, 117, "home_in"), (118, 118, "night_out"), (119, 134, "home_in"),
    (135, 142, "street_day"), (143, 164, "ball_out"), (165, 173, "song"), (174, 174, "title"),
    (175, 175, "school_in"), (176, 177, "school_out"), (178, 179, "home_in"),
]
TRUTH = {i: c for a, b, c in _RANGES for i in range(a, b + 1)}

QUIET = {"bedroom", "kitchen", "living_room", "shop", "neutral", ""}  # room tone / nothing: never wrong
OK = {
    "night_out": QUIET | {"night_street"},
    "home_in": QUIET,
    "school_in": QUIET | {"classroom", "school_corridor"},
    "school_out": QUIET | {"schoolyard", "park", "quiet_street"},
    "ball_out": QUIET | {"schoolyard", "park", "countryside", "quiet_street"},
    "street_day": QUIET | {"quiet_street", "city_street", "park"},
    "snow": QUIET,
    "dream": QUIET,
    "title": QUIET,
    "song": {""},  # credits: nothing at all
}


def wrong(idx: int, fill: str) -> bool:
    return fill not in OK[TRUTH[idx]]


def main() -> None:
    rows = list(csv.DictReader(open(sys.argv[1])))
    bad = [r for r in rows if wrong(int(r["idx"]), r["fill"])]
    secs = lambda rs: sum(float(r["end"]) - float(r["start"]) for r in rs)
    filled = [r for r in rows if r["fill"]]
    print(f"wrong: {len(bad)} of {len(filled)} filled pieces, {secs(bad):.0f} of {secs(filled):.0f} s")
    for r in bad:
        print(f"  {r['idx']:>3} {float(r['start']):7.1f}s {TRUTH[int(r['idx'])]:10s} got {r['fill']}")


if __name__ == "__main__":
    main()
