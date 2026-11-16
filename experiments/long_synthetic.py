"""T20: does memory stay flat when the soundtrack gets long?

Runs the memory-heavy middle of the pipeline on a synthetic mono signal of N minutes with
fake separators (no GPU, no video): stems written to files, voice rescue in place, gap
finding, fill rendered into a file and added, mix + --keep + WAV writing a block at a
time. Prints peak private commit and peak working set (Windows) after each stage.

    python experiments/long_synthetic.py MINUTES WORKDIR

Compare MINUTES=1 with MINUTES=120. WORKDIR needs about 4 GB of disk per hour (mono).
"""
from __future__ import annotations

import ctypes
import shutil
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import soundfile as sf

from musicremover.extract import WavArray
from musicremover.gaps import find_gaps
from musicremover.library import Library, Placement, add_fill, plan_fill, render_fill, split_songs
from musicremover.mix import keep_blocks, mix_blocks
from musicremover.separate import STEMS, FileArray, disk_stems, separate
from musicremover.separate.voice import rescue_voices

RATE = 48000
GIB = 1 << 30


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def memory() -> tuple[int, int, int]:
    """(private commit, working set, peak working set) of this process, in bytes."""
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PrivateUsage, c.WorkingSetSize, c.PeakWorkingSetSize


class Fake:
    sample_rate = RATE
    chunk_size = 10 * RATE

    def separate_chunk(self, chunk):
        return np.stack([0.3 * chunk, 0.6 * chunk, 0.02 * chunk])


class FakeVoice:
    sample_rate = RATE
    chunk_size = 10 * RATE

    def __init__(self):
        self.calls = 0

    def separate_chunk(self, chunk):
        self.calls += 1
        return np.stack([chunk, chunk * np.float32(10 ** ((-5, -22, -26, -45)[self.calls % 4] / 20))])


def main(minutes: float, work: Path) -> None:
    peak = {"commit": 0, "ws": 0}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            c, w, _ = memory()
            peak["commit"], peak["ws"] = max(peak["commit"], c), max(peak["ws"], w)
            time.sleep(0.1)

    threading.Thread(target=poll, daemon=True).start()
    base = memory()[0]
    work.mkdir(parents=True, exist_ok=True)
    n = int(minutes * 60 * RATE)
    t0 = time.time()

    def stage(name):
        print(f"{name:<18} peak commit {peak['commit'] / GIB:5.2f} GiB (+{(peak['commit'] - base) / GIB:5.2f}) "
              f"peak working set {peak['ws'] / GIB:5.2f} GiB  [{time.time() - t0:.0f}s]", flush=True)

    # the "soundtrack": 60 s of loud signal (speech + music) then 30 s of near silence
    rng = np.random.default_rng(0)
    with sf.SoundFile(work / "in.wav", "w", RATE, 1, subtype="FLOAT", format="WAV") as f:  # as extract() makes it
        for a in range(0, n, RATE * 10):
            b = min(n, a + RATE * 10)
            level = 0.1 if (a // RATE) % 90 < 60 else 1e-4
            block = (rng.standard_normal((1, b - a)) * level).astype(np.float32)
            if a <= n // 2 < b:
                block[:, n // 2 - a:n // 2 - a + 2000] += 1.5  # something for the limiter
            f.write(block.T)
    audio = WavArray(work / "in.wav")
    stage("signal")
    stems = separate(Fake(), audio, overlap=2, out=disk_stems(work, STEMS, audio.shape))
    stage("separate")
    (work / "voice").mkdir(exist_ok=True)
    stems = rescue_voices(stems, FakeVoice(), out=stems, scratch=work / "voice")
    stage("voice rescue")
    gaps = find_gaps(stems, RATE)
    gaps, _ = split_songs(gaps, np.full(1 + n // (RATE // 10), -30.0), n / RATE)
    stage("find gaps")
    lib = Library(work, {"park": (work / "park.wav",)})
    pieces = [(i, g) for i, g in enumerate(gaps)]
    placements, _ = plan_fill(pieces, ["park"] * len(pieces))
    fill = render_fill(placements, lib, 1, n, RATE,
                       read=lambda p, r, c: ((np.random.default_rng(1).standard_normal((1, 30 * r)) * 0.1).astype(np.float32), -20.0),
                       out=FileArray(work / "fill.f32", (1, n)))
    add_fill(stems["effects"], fill)
    del fill
    stage("fill")
    blocks = keep_blocks(mix_blocks(stems, -20.0, 0.99, RATE), audio,
                         [(600.0, 720.0), (minutes * 60 - 300, float("inf"))], RATE)
    with sf.SoundFile(work / "out.wav", "w", RATE, 1, subtype="FLOAT", format="WAV") as f:
        for _, block in blocks:
            f.write(np.ascontiguousarray(block.T))
    stage("mix and write")
    print(f"gaps {len(gaps)}, placements {len(placements)}, out.wav {(work / 'out.wav').stat().st_size / GIB:.2f} GiB")
    stop.set()
    audio.close()
    del audio, stems, blocks
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), Path(sys.argv[2]))
