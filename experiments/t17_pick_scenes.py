"""T17: pick talk-free gap stretches (5-15 s) with a clear scene label.

Usage:
    python experiments/t17_pick_scenes.py

Reads output/t15/gaps.csv, output/t18a/labels.csv and the speech stem of the
T1 scan (output/bakeoff/scan/speech.wav, read in blocks to keep memory low).
Prints, per gap, runs of pieces that share a label (close_up joins either
neighbour) and how much of each run has talking, so three scenes with
different labels can be chosen by hand.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "output" / "bakeoff" / "scan" / "speech.wav"
SKIP = [(129, 199), (1340, 1415)]  # opening and ending songs


def speech_levels() -> np.ndarray:
    """0.1 s speech level in dB, same as gaps.frame_levels but read in blocks."""
    out = []
    with sf.SoundFile(SPEECH) as f:
        frame = int(0.1 * f.samplerate)
        for block in f.blocks(blocksize=frame * 600, dtype="float32", always_2d=True):
            n = -(-len(block) // frame)
            pad = np.zeros((n * frame, block.shape[1]))
            pad[:len(block)] = block
            out.append((pad ** 2).reshape(n, frame, -1).mean(axis=(1, 2)))
    return 10 * np.log10(np.concatenate(out) + 1e-12)


def main() -> None:
    talk = speech_levels() > -45
    gaps = list(csv.DictReader(open(ROOT / "output" / "t15" / "gaps.csv")))
    pieces = list(csv.DictReader(open(ROOT / "output" / "t18a" / "labels.csv")))
    for g in gaps:
        gs, ge = float(g["start"]), float(g["end"])
        if any(gs < b and ge > a for a, b in SKIP):
            continue
        inside = [p for p in pieces if gs - 0.01 <= float(p["start"]) and float(p["end"]) <= ge + 0.01]
        # sliding windows over consecutive pieces: report stretches 5-15 s with little talk
        for i in range(len(inside)):
            for j in range(i, len(inside)):
                s, e = float(inside[i]["start"]), float(inside[j]["end"])
                if e - s > 15:
                    break
                if e - s < 5:
                    continue
                labels = {p["label"] for p in inside[i:j + 1]} - {"close_up"}
                if len(labels) != 1:
                    continue
                frac = float(np.mean(talk[int(s * 10):int(e * 10)]))
                if frac <= 0.15:
                    print(f"{s:7.1f}-{e:7.1f} ({e - s:4.1f} s) {labels.pop():13s} talk {frac:4.0%} "
                          f"music {g['music_db']} fx {g['effects_db']}  pieces {j - i + 1}")


if __name__ == "__main__":
    main()
