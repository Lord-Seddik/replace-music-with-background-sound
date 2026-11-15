"""Write the output video (the only module that writes video files)."""
from __future__ import annotations

import subprocess
from pathlib import Path

_AUDIO_CODECS = {
    ".mkv": "libopus",
    ".webm": "libopus",
    ".mp4": "aac",
    ".m4v": "aac",
    ".mov": "aac",
}
_MATROSKA = {".mkv", ".webm"}


def mux(
    original: Path,
    new_audio: Path,
    out: Path,
    *,
    keep_original: bool = False,
    language: str | None = None,
    title: str = "No music",
) -> Path:
    """Copy ``original`` with ``new_audio`` as its default audio track."""
    original, new_audio, out = Path(original), Path(new_audio), Path(out)
    ext = out.suffix.lower()
    codec = _AUDIO_CODECS.get(ext)
    if codec is None:
        raise ValueError(
            f"Unsupported container {ext!r}; supported: "
            f"{', '.join(sorted(_AUDIO_CODECS))}"
        )
    if out.exists():
        if original.exists() and out.resolve() == original.resolve():
            raise ValueError("Output path is the same file as the original")
        raise FileExistsError(f"Output already exists: {out}")

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-n",
           "-i", str(original), "-i", str(new_audio),
           "-map", "0:v?", "-map", "1:a:0"]
    if keep_original:
        cmd += ["-map", "0:a?"]
    cmd += ["-map", "0:s?"]
    if ext in _MATROSKA:
        cmd += ["-map", "0:t?"]
    cmd += ["-map_chapters", "0", "-map_metadata", "0",
            "-c", "copy",
            "-c:a:0", codec, "-b:a:0", "192k",
            "-disposition:a", "0", "-disposition:a:0", "default",
            "-metadata:s:a:0", f"title={title}"]
    if language:
        cmd += ["-metadata:s:a:0", f"language={language}"]
    cmd.append(str(out))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    if proc.returncode != 0:
        if out.exists():
            out.unlink()  # partial file: it did not exist before
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out
