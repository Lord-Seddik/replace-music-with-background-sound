import hashlib
import json
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from musicremover.cli import main


class FakeSeparator:
    sample_rate = 48000
    chunk_size = 4800

    def separate_chunk(self, chunk):
        return np.stack([0.5 * chunk, 0.5 * chunk, np.zeros_like(chunk)])


def factory(weights, device):
    return FakeSeparator()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audio_streams(p):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-print_format", "json",
         "-show_streams", str(p)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


@pytest.fixture
def src(media, tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    f = d / "Episode 01.mkv"
    shutil.copy(media, f)
    return f


def test_single_file(src, capsys):
    before = sha(src)
    assert main([str(src)], factory) == 0
    out = src.parent / "Episode 01.nomusic.mkv"
    assert out.is_file()
    streams = audio_streams(out)
    assert len(streams) == 1
    assert streams[0]["disposition"]["default"] == 1
    assert streams[0]["tags"]["title"] == "No music"
    assert sha(src) == before
    assert str(out) in capsys.readouterr().out


def test_folder(src, media):
    shutil.copy(media, src.parent / "Episode 02.mkv")
    shutil.copy(media, src.parent / "Episode 03.nomusic.mkv")
    (src.parent / "notes.txt").write_text("x")
    assert main([str(src.parent)], factory) == 0
    names = sorted(p.name for p in src.parent.iterdir())
    assert "Episode 01.nomusic.mkv" in names
    assert "Episode 02.nomusic.mkv" in names
    assert "Episode 03.nomusic.nomusic.mkv" not in names
    assert len([n for n in names if ".nomusic." in n]) == 3


def test_existing_output_skipped(src, capsys):
    out = src.parent / "Episode 01.nomusic.mkv"
    out.write_bytes(b"keep me")
    assert main([str(src)], factory) == 0
    assert out.read_bytes() == b"keep me"
    assert "skipped (exists)" in capsys.readouterr().out


def test_skip_does_not_build_separator(src):
    (src.parent / "Episode 01.nomusic.mkv").write_bytes(b"x")

    def boom(weights, device):
        raise AssertionError("separator built")

    assert main([str(src)], boom) == 0


def test_out_dir(src, tmp_path):
    od = tmp_path / "out" / "nested"
    assert main([str(src), "-o", str(od)], factory) == 0
    assert (od / "Episode 01.nomusic.mkv").is_file()
    assert not (src.parent / "Episode 01.nomusic.mkv").exists()


def test_keep_original(src, tmp_path):
    assert main([str(src), "-o", str(tmp_path / "a")], factory) == 0
    assert main([str(src), "-o", str(tmp_path / "b"), "--keep-original"], factory) == 0
    assert len(audio_streams(tmp_path / "a" / "Episode 01.nomusic.mkv")) == 1
    kept = audio_streams(tmp_path / "b" / "Episode 01.nomusic.mkv")
    assert len(kept) == 3
    assert kept[0]["tags"]["title"] == "No music"


def test_language_uses_51_track(src):
    assert main([str(src), "--language", "eng"], factory) == 0
    streams = audio_streams(src.parent / "Episode 01.nomusic.mkv")
    assert streams[0]["channels"] == 2


def test_missing_language_fails_exit_1(src, media, capsys):
    other = src.parent / "Episode 02.mkv"
    shutil.copy(media, other)
    assert main([str(src), str(other), "--language", "fra"], factory) == 1
    err = capsys.readouterr().err
    assert err.count("failed:") == 2 and "fra" in err
    assert not list(src.parent.glob("*.nomusic.*"))


def test_failure_does_not_stop_others(src, video_only, capsys):
    bad = src.parent / "Episode 00.mkv"  # sorts first, has no audio
    shutil.copy(video_only, bad)
    assert main([str(src.parent)], factory) == 1
    assert "failed:" in capsys.readouterr().err
    assert (src.parent / "Episode 01.nomusic.mkv").is_file()
    assert not (src.parent / "Episode 00.nomusic.mkv").exists()


def test_missing_weights(src, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MUSICREMOVER_WEIGHTS", raising=False)
    code = main([str(src), "--weights", str(tmp_path / "nope.ckpt")])
    assert code == 2
    assert "zenodo" in capsys.readouterr().err
    assert not (src.parent / "Episode 01.nomusic.mkv").exists()


def test_no_inputs_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)], factory) == 2


def test_no_temp_files_left(src, tmp_path, monkeypatch):
    t = tmp_path / "tmpdir"
    t.mkdir()
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(t))
    monkeypatch.setattr(tempfile, "tempdir", str(t))
    assert main([str(src), "--language", "fra"], factory) == 1
    assert list(t.iterdir()) == []
    assert main([str(src)], factory) == 0
    assert list(t.iterdir()) == []


def test_interrupt_cleans_up(src, tmp_path, monkeypatch):
    t = tmp_path / "tmpdir"
    t.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(t))

    class Bad(FakeSeparator):
        def separate_chunk(self, chunk):
            raise KeyboardInterrupt

    assert main([str(src)], lambda w, d: Bad()) == 130
    assert list(t.iterdir()) == []
    assert not (src.parent / "Episode 01.nomusic.mkv").exists()


def _decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_keep_range_keeps_original_audio(src):
    assert main([str(src), "--keep", "0-1"], factory) == 0
    out = _decode(src.parent / "Episode 01.nomusic.mkv")
    orig = _decode(src)
    n = min(len(out), len(orig))
    a, b = out[2000:46000], orig[2000:46000]
    assert np.corrcoef(a, b)[0, 1] > 0.9
    assert abs(20 * np.log10(_rms(a) / _rms(b))) < 1.0
    s = slice(int(1.1 * 48000), n)
    ratio_db = 20 * np.log10(_rms(out[s]) / _rms(orig[s]))
    assert abs(ratio_db - (-6.02)) < 1.5


def test_keep_bad_value_exits_2(src, capsys):
    with pytest.raises(SystemExit) as e:
        main([str(src), "--keep", "banana"], factory)
    assert e.value.code == 2
    assert "banana" in capsys.readouterr().err


class FakeVoice:
    """Calls the whole music stem a lone voice, so all of it is rescued."""

    sample_rate = 48000
    chunk_size = 4800

    def separate_chunk(self, chunk):
        return np.stack([chunk, np.zeros_like(chunk)])


def test_voice_rescue_puts_music_stem_voice_back(src):
    # fake separator: speech 0.5x, music 0.5x; rescuing all music gives ~1.0x
    assert main([str(src)], factory, lambda device: FakeVoice()) == 0
    out, orig = _decode(src.parent / "Episode 01.nomusic.mkv"), _decode(src)
    n = min(len(out), len(orig))
    assert abs(20 * np.log10(_rms(out[4800:n - 4800]) / _rms(orig[4800:n - 4800]))) < 1.0


def test_no_voice_rescue_skips_the_voice_backend(src):
    def boom(device):
        raise AssertionError("voice backend must not be built")

    assert main([str(src), "--no-voice-rescue"], factory, boom) == 0


class FakeScenes:
    """Calls every still a park."""

    def scores(self, images, texts):
        return np.array([[5.0 if "park" in t else 0.0 for t in texts]] * len(images))


@pytest.fixture
def library(tmp_path):
    import soundfile as sf
    d = tmp_path / "lib" / "park"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(d / "birds.wav", (0.1 * rng.standard_normal(48000 * 2)).astype(np.float32), 48000)
    return tmp_path / "lib"


def test_fill_library_lays_ambience_under_the_gap(src, library, tmp_path, capsys):
    # fake separator: the whole clip is music-dominated (effects silent), so it is one gap
    assert main([str(src), "-o", str(tmp_path / "plain")], factory) == 0
    assert main([str(src), "-o", str(tmp_path / "fill"), "--fill", "library", "--library", str(library)],
                factory, scene_factory=lambda device: FakeScenes()) == 0
    plain = _decode(tmp_path / "plain" / "Episode 01.nomusic.mkv")
    filled = _decode(tmp_path / "fill" / "Episode 01.nomusic.mkv")
    n = min(len(plain), len(filled))
    diff_db = 20 * np.log10(_rms(filled[48000:n - 48000] - plain[48000:n - 48000]))
    assert -46 < diff_db < -34  # the fill, at about -40 dB
    assert "ambience laid under 0:03 in 1 place" in capsys.readouterr().out


def test_fill_level_shifts_the_fill(src, library, tmp_path):
    base = ["--fill", "library", "--library", str(library)]
    assert main([str(src), "-o", str(tmp_path / "plain")], factory) == 0
    assert main([str(src), "-o", str(tmp_path / "quiet"), *base, "--fill-level", "-6"],
                factory, scene_factory=lambda device: FakeScenes()) == 0
    plain = _decode(tmp_path / "plain" / "Episode 01.nomusic.mkv")
    quiet = _decode(tmp_path / "quiet" / "Episode 01.nomusic.mkv")
    n = min(len(plain), len(quiet))
    diff_db = 20 * np.log10(_rms(quiet[48000:n - 48000] - plain[48000:n - 48000]))
    assert -52 < diff_db < -40  # about -46 dB: 6 dB below the default -40


@pytest.mark.parametrize("args, message", [
    (["--fill-level", "3"], "only used with --fill library"),
    (["--fill", "library", "--library", "x", "--fill-level", "25"], "between -20 and 20"),
    (["--fill", "library"], "needs --library"),
    (["--library", "x"], "only used with --fill library"),
    (["--fill", "library", "--library", "missing-folder"], "not found"),
])
def test_fill_bad_options_exit_2(src, args, message, capsys):
    assert main([str(src), *args], factory) == 2
    assert message in capsys.readouterr().err
