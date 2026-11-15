import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from musicremover.mux import mux
from musicremover.probe import probe


def _ff(args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


def _streams(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_chapters", str(path)],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    return d["streams"], d["chapters"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _by_type(streams, kind):
    return [s for s in streams if s["codec_type"] == kind]


@pytest.fixture(scope="session")
def new_wav(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("wav") / "new.wav"
    _ff(["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=3",
         "-ac", "2", "-c:a", "pcm_f32le", str(out)])
    return out


@pytest.fixture(scope="session")
def rich_mkv(tmp_path_factory) -> Path:
    """mkv with a font attachment, two chapters, video, audio, subtitle."""
    d = tmp_path_factory.mktemp("rich")
    srt = d / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    font = d / "f.ttf"
    font.write_bytes(b"\x00\x01\x00\x00fakefontdata")
    meta = d / "meta.txt"
    meta.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1500\ntitle=One\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1500\nEND=3000\ntitle=Two\n",
        encoding="utf-8")
    out = d / "rich.mkv"
    _ff(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
         "-i", str(srt), "-i", str(meta),
         "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3",
         "-map_chapters", "3",
         "-c:v", "libx264", "-c:a", "aac", "-c:s", "srt",
         "-attach", str(font),
         "-metadata:s:t", "mimetype=application/x-truetype-font",
         "-metadata:s:t", "filename=f.ttf", str(out)])
    return out


@pytest.fixture(scope="session")
def mp4_in(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("mp4") / "in.mp4"
    _ff(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
         "-c:v", "libx264", "-c:a", "aac", str(out)])
    return out


def test_layout_and_codecs(media, new_wav, tmp_path):
    out = mux(media, new_wav, tmp_path / "o.mkv")
    assert out == tmp_path / "o.mkv" and out.is_file()
    streams, _ = _streams(out)
    orig, _ = _streams(media)
    assert [s["codec_type"] for s in streams] == ["video", "audio", "subtitle"]
    assert streams[0]["codec_name"] == _by_type(orig, "video")[0]["codec_name"]
    a = streams[1]
    assert a["codec_name"] == "opus"
    assert a["disposition"]["default"] == 1
    assert a["tags"]["title"] == "No music"
    assert streams[2]["codec_name"] == "subrip"


def test_language_and_title(media, new_wav, tmp_path):
    out = mux(media, new_wav, tmp_path / "o.mkv", language="jpn", title="Clean")
    a = _by_type(_streams(out)[0], "audio")[0]
    assert a["tags"]["language"] == "jpn"
    assert a["tags"]["title"] == "Clean"


def test_keep_original(media, new_wav, tmp_path):
    out = mux(media, new_wav, tmp_path / "o.mkv", keep_original=True)
    streams, _ = _streams(out)
    audio = _by_type(streams, "audio")
    assert [s["codec_name"] for s in audio] == ["opus", "aac", "aac"]
    assert [s["disposition"]["default"] for s in audio] == [1, 0, 0]
    assert audio[1]["tags"]["language"] == "jpn"
    assert audio[2]["tags"]["language"] == "eng"
    assert audio[2]["channels"] == 6
    assert len(_by_type(streams, "subtitle")) == 1
    info = probe(out)
    assert [t.default for t in info.audio] == [True, False, False]


def test_attachments_and_chapters_survive(rich_mkv, new_wav, tmp_path):
    out = mux(rich_mkv, new_wav, tmp_path / "o.mkv")
    streams, chapters = _streams(out)
    att = _by_type(streams, "attachment")
    assert len(att) == 1
    assert att[0]["tags"]["mimetype"] == "application/x-truetype-font"
    assert len(chapters) == 2
    assert [c["tags"]["title"] for c in chapters] == ["One", "Two"]
    assert len(_by_type(streams, "subtitle")) == 1


def test_mp4_uses_aac(mp4_in, new_wav, tmp_path):
    out = mux(mp4_in, new_wav, tmp_path / "o.mp4")
    streams, _ = _streams(out)
    audio = _by_type(streams, "audio")
    assert len(audio) == 1 and audio[0]["codec_name"] == "aac"
    assert audio[0]["disposition"]["default"] == 1
    assert _by_type(streams, "video")[0]["codec_name"] == "h264"


def test_durations_match(media, new_wav, tmp_path):
    out = mux(media, new_wav, tmp_path / "o.mkv")
    assert abs(probe(out).duration - probe(media).duration) < 0.1


def test_original_unchanged(media, new_wav, tmp_path):
    before = _sha(media)
    mux(media, new_wav, tmp_path / "o.mkv", keep_original=True)
    assert _sha(media) == before


def test_existing_output_refused(media, new_wav, tmp_path):
    out = tmp_path / "o.mkv"
    out.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        mux(media, new_wav, out)
    assert out.read_bytes() == b"keep me"


def test_same_file_refused(media, new_wav):
    before = _sha(media)
    with pytest.raises(ValueError):
        mux(media, new_wav, media)
    assert _sha(media) == before


def test_unsupported_extension(media, new_wav, tmp_path):
    with pytest.raises(ValueError, match="mkv"):
        mux(media, new_wav, tmp_path / "o.avi")


def test_failure_leaves_no_partial(media, tmp_path):
    out = tmp_path / "o.mkv"
    with pytest.raises(RuntimeError):
        mux(media, tmp_path / "missing.wav", out)
    assert not out.exists()
