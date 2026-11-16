"""T18a: label the scene of every gap piece from video stills, for checking by eye.

Usage:
    python experiments/label_gaps.py samples/episode.mkv

Uses the full-episode stems from the T1 scan (output/bakeoff/scan/) to find
the gaps (as in T15), splits them at scene cuts, grabs three stills per piece,
labels each piece with SigLIP and writes output/t18a/labels.csv plus a
contact sheet output/t18a/index.html (open it in a browser).
"""
import csv
import html
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from musicremover.gaps import find_gaps, split_at_cuts  # noqa: E402
from musicremover.scenes import (  # noqa: E402
    DEFAULT_SCENES, SiglipScenes, grab_frame, label_scenes, scene_cuts, still_times,
)

SCAN = ROOT / "output" / "bakeoff" / "scan"
OUT = ROOT / "output" / "t18a"


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def save_jpg(img: np.ndarray, path: Path) -> None:
    h, w, _ = img.shape
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-i", "-", "-q:v", "3", str(path)], input=img.tobytes(), check=True)


def main() -> None:
    episode = Path(sys.argv[1])
    (OUT / "stills").mkdir(parents=True, exist_ok=True)
    stems = {n: sf.read(SCAN / f"{n}.wav", dtype="float32", always_2d=True)[0].T.copy()
             for n in ("music", "effects")}
    t0 = time.monotonic()
    pieces = split_at_cuts(find_gaps(stems, 48000), scene_cuts(episode))
    del stems
    print(f"{len(pieces)} pieces ({time.monotonic() - t0:.0f} s for gaps and cuts)")

    cache = OUT / "stills.npz"  # stills take ~3 min to grab; keep them for re-runs
    if cache.is_file():
        with np.load(cache) as z:
            flat, starts = z["stills"], z["starts"]
        if len(starts) != len(pieces) or not np.allclose(starts, [p.start for p in pieces]):
            sys.exit(f"{cache} is for other pieces; delete it")
        stills = [list(flat[3 * i:3 * i + 3]) for i in range(len(pieces))]
    else:
        t0 = time.monotonic()
        stills = [[grab_frame(episode, t) for t in still_times(p.start, p.end)] for p in pieces]
        print(f"{sum(map(len, stills))} stills in {time.monotonic() - t0:.0f} s")
        np.savez(cache, stills=np.stack([s for g in stills for s in g]),
                 starts=np.array([p.start for p in pieces]))

    t0 = time.monotonic()
    model = SiglipScenes()
    print(f"model loaded in {time.monotonic() - t0:.0f} s")
    t0 = time.monotonic()
    labels = label_scenes(stills, model)
    print(f"labelled in {time.monotonic() - t0:.1f} s")

    cards = []
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "label", "score", "runner_up", "runner_up_score"])
        for i, (p, lb, group) in enumerate(zip(pieces, labels, stills)):
            w.writerow([p.start, p.end, lb.label, lb.score, lb.runner_up, lb.runner_up_score])
            name = f"{i:03d}.jpg"
            save_jpg(group[len(group) // 2], OUT / "stills" / name)
            cards.append(
                f'<figure><img src="stills/{name}" loading="lazy" alt="">'
                f"<figcaption><b>{html.escape(lb.label)}</b> {lb.score:.0%}"
                f"<span>{clock(p.start)}–{clock(p.end)} · then {html.escape(lb.runner_up)} "
                f"{lb.runner_up_score:.0%}</span></figcaption></figure>")
    (OUT / "index.html").write_text(PAGE.format(
        count=len(pieces), cards="\n".join(cards),
        scenes=", ".join(DEFAULT_SCENES)), encoding="utf-8")

    for name, n in Counter(lb.label for lb in labels).most_common():
        print(f"  {name:16} {n}")
    print(f"open {OUT / 'index.html'}")


PAGE = """<!doctype html><meta charset="utf-8"><title>T18a scene labels</title>
<style>
body{{font:14px system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
img{{width:100%;display:block}} figcaption{{padding:6px 8px}}
figcaption span{{display:block;color:#666;font-size:12px}}
</style>
<h1>Scene labels for {count} gap pieces</h1>
<p>Middle still of each piece. Choices: {scenes}.</p>
<main>
{cards}
</main>
"""

if __name__ == "__main__":
    main()
