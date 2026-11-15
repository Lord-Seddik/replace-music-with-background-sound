"""Inspect media files with ffprobe (the only module that calls ffprobe)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProbeError(RuntimeError):
    """Raised when a file cannot be probed or has no usable audio."""


@dataclass(frozen=True)
class AudioTrack:
    """One audio stream of a media file."""

    index: int
    position: int
    codec: str
    channels: int
    sample_rate: int
    language: str | None
    title: str | None
    default: bool


@dataclass(frozen=True)
class MediaInfo:
    """Summary of a media file."""

    path: Path
    duration: float
    audio: tuple[AudioTrack, ...]
    has_video: bool
    subtitle_count: int


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def probe(path: str | Path) -> MediaInfo:
    """Probe ``path`` and return its audio tracks and basic facts."""
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"File not found: {path}")
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe not found on PATH") from exc
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {path}") from exc

    audio: list[AudioTrack] = []
    has_video = False
    subtitles = 0
    stream_durations: list[float] = []
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        disposition = stream.get("disposition") or {}
        d = _float(stream.get("duration"))
        if d is not None:
            stream_durations.append(d)
        if kind == "audio":
            tags = {k.lower(): v for k, v in (stream.get("tags") or {}).items()}
            audio.append(AudioTrack(
                index=int(stream["index"]),
                position=len(audio),
                codec=str(stream.get("codec_name", "unknown")),
                channels=int(stream.get("channels", 0)),
                sample_rate=int(stream.get("sample_rate", 0)),
                language=tags.get("language"),
                title=tags.get("title"),
                default=bool(disposition.get("default", 0)),
            ))
        elif kind == "video":
            if not disposition.get("attached_pic", 0):
                has_video = True
        elif kind == "subtitle":
            subtitles += 1

    if not audio:
        raise ProbeError(f"No audio track found in {path}")

    duration = _float((data.get("format") or {}).get("duration"))
    if duration is None:
        duration = max(stream_durations, default=0.0)
    return MediaInfo(path=path, duration=duration, audio=tuple(audio),
                     has_video=has_video, subtitle_count=subtitles)


def choose_track(info: MediaInfo, language: str | None = None) -> AudioTrack:
    """Pick a track by language, else the default-flagged one, else the first."""
    if language is not None:
        for track in info.audio:
            if track.language == language:
                return track
        available = sorted({t.language or "und" for t in info.audio})
        raise ProbeError(
            f"No audio track with language {language!r}; "
            f"available: {', '.join(available)}"
        )
    for track in info.audio:
        if track.default:
            return track
    return info.audio[0]
