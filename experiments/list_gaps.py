"""T15: list the gaps for an episode and cut a few to listen to.

Usage:
    python experiments/list_gaps.py samples/episode.mkv output/t13/episode/episode.nomusic.mkv

Uses the full-episode stems from the T1 scan (output/bakeoff/scan/) for the
levels, detects scene cuts from the video, prints the gap list, writes it to
output/t15/gaps.csv, and cuts three gaps from the music-free episode
(original video, new sound) into output/t15/ for listening.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from musicremover.gaps import find_gaps, frame_levels, split_at_cuts  # noqa: E402
from musicremover.scenes import scene_cuts  # noqa: E402

SCAN = ROOT / "output" / "bakeoff" / "scan"
OUT = ROOT / "output" / "t15"


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main() -> None:
    episode, nomusic = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    stems = {n: sf.read(SCAN / f"{n}.wav", dtype="float32", always_2d=True)[0].T.copy()
             for n in ("speech", "music", "effects")}
    gaps = find_gaps(stems, 48000)
    cuts = scene_cuts(episode)
    pieces = split_at_cuts(gaps, cuts)
    speech = frame_levels(stems["speech"], 48000)

    total = sum(g.length for g in gaps)
    print(f"{len(cuts)} scene cuts; {len(gaps)} gaps, {total / 60:.1f} min; {len(pieces)} pieces after splitting at cuts")
    rows = []
    for g in gaps:
        talk = float(np.mean(speech[int(g.start * 10):int(g.end * 10)] > -45))
        shots = sum(1 for p in pieces if g.start <= p.start < g.end)
        rows.append((g, talk, shots))
        print(f"  {clock(g.start):>7} - {clock(g.end):>7}  {g.length:5.1f} s  music {g.music_db:6.1f} dB  "
              f"background {g.effects_db:6.1f} dB  talking {talk:4.0%}  shots {shots}")
    with open(OUT / "gaps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "music_db", "effects_db", "talking", "shots"])
        for g, talk, shots in rows:
            w.writerow([g.start, g.end, g.music_db, g.effects_db, round(talk, 2), shots])

    # three to listen to: the longest mostly-silent one, the longest with talk, and a mid one
    quiet = max((r for r in rows if r[1] < 0.3), key=lambda r: r[0].length, default=None)
    talky = max((r for r in rows if r[1] >= 0.5), key=lambda r: r[0].length, default=None)
    rest = sorted((r for r in rows if r not in (quiet, talky)), key=lambda r: r[0].length)
    middle = rest[len(rest) // 2] if rest else None
    for name, r in (("1_no_talk", quiet), ("2_talk", talky), ("3_typical", middle)):
        if r is None:
            continue
        g = r[0]
        out = OUT / f"{name}_{clock(g.start).replace(':', '-')}.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", str(max(0, g.start - 2)),
                        "-t", str(min(g.length, 30) + 4), "-i", str(nomusic), "-map", "0:v:0", "-map", "0:a:0",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out)],
                       check=True)
        print(f"listen: {out.name}")


if __name__ == "__main__":
    main()
