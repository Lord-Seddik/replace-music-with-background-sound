"""T22: stems and the original are read/written with plain file calls, not memory maps."""
import numpy as np
import pytest
import soundfile as sf

from musicremover.extract import WavArray
from musicremover.mix import keep_blocks, keep_ranges, mix, mix_blocks
from musicremover.separate import STEMS, FileArray, disk_stems, separate
from musicremover.separate.voice import rescue_voices

RATE = 48000


class Split:
    sample_rate = RATE
    chunk_size = 3000

    def separate_chunk(self, chunk):
        return np.stack([0.5 * chunk, 0.3 * chunk, 0.2 * chunk * chunk])


def noise(ch, n, seed=0):
    return (np.random.default_rng(seed).standard_normal((ch, n)) * 0.3).astype(np.float32)


def test_file_array_reads_what_was_written(tmp_path):
    x = noise(2, 1000)
    a = FileArray(tmp_path / "a.f32", (2, 1000))
    assert a.shape == (2, 1000) and a.ndim == 2 and a.dtype == np.float32
    assert (a[:, :] == 0).all()
    a[:, 100:400] = x[:, 100:400]
    np.testing.assert_array_equal(a[:, 100:400], x[:, 100:400])
    np.testing.assert_array_equal(a[:, 100:400][1], x[1, 100:400])
    a[:, 400:] = x[:, 400:]
    a[:, :100] = x[:, :100]
    np.testing.assert_array_equal(np.asarray(a), x)
    np.testing.assert_array_equal(a[1:2, 990:2000], x[1:2, 990:])  # past the end is clipped
    a[:, 0:10] += 1.0
    np.testing.assert_array_equal(a[:, 0:10], x[:, 0:10] + 1.0)
    with pytest.raises(TypeError):
        a[0]
    a.delete()
    assert not (tmp_path / "a.f32").exists()


def test_wav_array_matches_file(tmp_path):
    x = noise(2, 5000)
    sf.write(tmp_path / "in.wav", x.T, RATE, subtype="FLOAT")
    w = WavArray(tmp_path / "in.wav")
    assert w.shape == (2, 5000) and w.ndim == 2
    np.testing.assert_array_equal(w[:, 1234:4321], x[:, 1234:4321])
    np.testing.assert_array_equal(w[:, 0:10], x[:, 0:10])
    np.testing.assert_array_equal(np.asarray(w), x)
    w.close()
    (tmp_path / "in.wav").unlink()  # closed, so Windows lets it go


def test_separate_to_file_stems_equals_memory(tmp_path):
    x = noise(2, 20000, 1)
    want = separate(Split(), x, overlap=2)
    got = separate(Split(), x, overlap=2, out=disk_stems(tmp_path, STEMS, x.shape))
    for n in STEMS:
        assert isinstance(got[n], FileArray)
        np.testing.assert_array_equal(np.asarray(got[n]), want[n])


def test_separate_reading_the_original_from_a_wav(tmp_path):
    x = noise(2, 20000, 2)
    sf.write(tmp_path / "in.wav", x.T, RATE, subtype="FLOAT")
    w = WavArray(tmp_path / "in.wav")
    want = separate(Split(), x, overlap=2)
    got = separate(Split(), w, overlap=2)
    for n in STEMS:
        np.testing.assert_array_equal(got[n], want[n])
    w.close()


class FakeVoice:
    sample_rate = RATE
    chunk_size = 5000

    def separate_chunk(self, chunk):
        return np.stack([chunk, chunk * np.float32(1e-3)])


def test_voice_rescue_in_place_on_file_stems_equals_memory(tmp_path):
    stems = {n: noise(2, 12000, i) for i, n in enumerate(STEMS)}
    want = rescue_voices({k: v.copy() for k, v in stems.items()}, FakeVoice())
    files = disk_stems(tmp_path, STEMS, (2, 12000))
    for n in STEMS:
        files[n][:, :] = stems[n]
    (tmp_path / "v").mkdir()
    got = rescue_voices(files, FakeVoice(), out=files, scratch=tmp_path / "v")
    for n in STEMS:
        np.testing.assert_array_equal(np.asarray(got[n]), want[n])
    assert not list((tmp_path / "v").iterdir())  # scratch freed


@pytest.mark.parametrize("gain", [None, -12.0])
def test_mix_and_keep_from_file_arrays_equal_memory(tmp_path, gain):
    n = 3 * RATE
    stems = {k: noise(2, n, i) * (4 if k == "speech" else 1) for i, k in enumerate(STEMS)}
    orig = noise(2, n, 9)
    files = disk_stems(tmp_path, STEMS, (2, n))
    for k in STEMS:
        files[k][:, :] = stems[k]
    sf.write(tmp_path / "in.wav", orig.T, RATE, subtype="FLOAT")
    w = WavArray(tmp_path / "in.wav")
    ranges = [(0.5, 1.0), (2.5, float("inf"))]
    want = keep_ranges(mix(stems, gain, 0.99, RATE), orig, ranges, RATE)
    out = np.empty((2, n), np.float32)
    for a, y in keep_blocks(mix_blocks(files, gain, 0.99, RATE, 20000), w, ranges, RATE):
        out[:, a:a + y.shape[1]] = y
    np.testing.assert_allclose(out, want, atol=2e-6, rtol=0)
    assert mix(files, gain, 0.99, RATE).shape == (2, n)
    w.close()
