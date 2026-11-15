"""T1 throwaway: cut one-minute test clips from an episode.

Usage:
    python experiments/make_clips.py samples/episode.mkv            # 3 clips at 25/50/75 %
    python experiments/make_clips.py samples/episode.mkv 4:10 11:30 17:05
    python experiments/make_clips.py --first 4 samples/episode.mkv 2:15   # numbering from clip4

Writes output/bakeoff/clips/clipN.wav (48 kHz stereo) and clipN.mkv (with
video, for watching). The episode itself is only read.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "bakeoff" / "clips"
LENGTH = 60.0


def seconds(text: str) -> float:
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def duration(path: Path) -> float:
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(info.stdout)["format"]["duration"])


def main() -> None:
    args = sys.argv[1:]
    first = 1
    if args[:1] == ["--first"]:
        first, args = int(args[1]), args[2:]
    if not args:
        sys.exit(__doc__)
    episode = Path(args[0]).resolve()
    if args[1:]:
        starts = [seconds(t) for t in args[1:]]
    else:
        total = duration(episode)
        starts = [total * f for f in (0.25, 0.5, 0.75)]

    OUT.mkdir(parents=True, exist_ok=True)
    for n, start in enumerate(starts, first):
        base = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", str(LENGTH), "-i", str(episode)]
        subprocess.run(base + ["-map", "0:a:0", "-ac", "2", "-ar", "48000", "-c:a", "pcm_f32le",
                               str(OUT / f"clip{n}.wav")], check=True)
        subprocess.run(base + ["-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
                               "-crf", "23", "-c:a", "aac", str(OUT / f"clip{n}.mkv")], check=True)
        print(f"clip{n}: starts at {int(start // 60)}:{start % 60:04.1f}")


if __name__ == "__main__":
    main()
