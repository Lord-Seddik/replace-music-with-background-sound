"""Extract an audio track to WAV with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .probe import AudioTrack


def extract(path: str | Path, track: AudioTrack, out_wav: str | Path,
            sample_rate: int = 48000) -> Path:
    """Write ``track`` as 32-bit float stereo WAV; never overwrites ``out_wav``."""
    out_wav = Path(out_wav)
    if out_wav.exists():
        raise FileExistsError(f"Output already exists: {out_wav}")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-n",
        "-i", str(path),
        "-map", f"0:a:{track.position}",
        "-vn", "-sn",
        "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out_wav
