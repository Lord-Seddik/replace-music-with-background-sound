import numpy as np
import pytest
import soundfile as sf

from musicremover.extract import extract
from musicremover.probe import AudioTrack, probe


@pytest.mark.parametrize("position", [0, 1])
def test_extract(media, tmp_path, position):
    track = probe(media).audio[position]
    out = extract(media, track, tmp_path / "o.wav")
    data, sr = sf.read(out, dtype="float32")
    assert sf.info(out).subtype == "FLOAT"
    assert sr == 48000
    assert data.ndim == 2 and data.shape[1] == 2
    assert len(data) / sr == pytest.approx(3.0, abs=0.05)
    assert np.abs(data).max() > 0.05


def test_existing_output(media, tmp_path):
    out = tmp_path / "o.wav"
    out.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        extract(media, probe(media).audio[0], out)
    assert out.read_bytes() == b"keep"


def test_ffmpeg_failure(tmp_path):
    t = AudioTrack(1, 0, "aac", 2, 48000, None, None, True)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        extract(tmp_path / "missing.mkv", t, tmp_path / "o.wav")
