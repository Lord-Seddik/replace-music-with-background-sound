import subprocess
from pathlib import Path

import pytest


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> Path:
    """3 s mkv: video, stereo 48k jpn (default), 5.1 44.1k eng, SRT subtitle."""
    d = tmp_path_factory.mktemp("media")
    srt = d / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    out = d / "multi.mkv"
    _ff([
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100:duration=3",
        "-i", str(srt),
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
        "-c:v", "libx264", "-c:a:0", "aac", "-c:a:1", "aac", "-c:s", "srt",
        "-ac:a:0", "2", "-ac:a:1", "6", "-ar:a:0", "48000", "-ar:a:1", "44100",
        "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=eng",
        "-disposition:a:0", "default", "-disposition:a:1", "0",
        str(out),
    ])
    return out


@pytest.fixture(scope="session")
def video_only(tmp_path_factory) -> Path:
    """3 s mkv with no audio stream."""
    out = tmp_path_factory.mktemp("noaudio") / "silent.mkv"
    _ff(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
         "-c:v", "libx264", str(out)])
    return out
