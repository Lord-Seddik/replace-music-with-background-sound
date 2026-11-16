import tracemalloc

import numpy as np
import pytest
import torch

from musicremover.separate import STEMS, disk_stems, separate
from musicremover.separate.bandit import match_keys


class Fake:
    """Puts the input in 'speech', a scaled copy in 'music', zeros in 'effects'."""

    sample_rate = 48000

    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size
        self.calls = 0

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        assert chunk.shape[1] == self.chunk_size
        self.calls += 1
        return np.stack([chunk, 0.5 * chunk, np.zeros_like(chunk)])


class Biased(Fake):
    """Adds a different constant in every chunk: seams would show as steps."""

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        out = super().separate_chunk(chunk)
        out[2] += self.calls
        return out


def noise(channels: int, length: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((channels, length)).astype(np.float32)


@pytest.mark.parametrize("length", [1, 999, 1000, 1001, 4321, 10_000])
@pytest.mark.parametrize("overlap", [1, 2, 4])
def test_joins_are_seamless_and_length_is_exact(length, overlap):
    audio = noise(2, length)
    stems = separate(Fake(), audio, overlap=overlap)
    assert list(stems) == list(STEMS)
    for stem in stems.values():
        assert stem.shape == audio.shape
        assert stem.dtype == np.float32
    np.testing.assert_allclose(stems["speech"], audio, atol=1e-5)
    np.testing.assert_allclose(stems["music"], 0.5 * audio, atol=1e-5)
    np.testing.assert_allclose(stems["effects"], 0, atol=1e-6)


def test_blending_is_smooth_between_chunks():
    stems = separate(Biased(), np.zeros((1, 5000), np.float32), overlap=2)
    steps = np.abs(np.diff(stems["effects"][0]))
    assert steps.max() < 0.01  # a hard seam would jump by 1.0


@pytest.mark.parametrize("overlap", [1, 1.1, 2, 4])
def test_stems_written_to_files_equal_stems_in_memory(tmp_path, overlap):
    audio = noise(2, 4321)
    expected = separate(Biased(), audio, overlap=overlap)
    out = disk_stems(tmp_path, STEMS, audio.shape)
    result = separate(Biased(), audio, overlap=overlap, out=out)
    assert result is out
    for name in STEMS:
        assert (tmp_path / f"{name}.f32").is_file()
        np.testing.assert_array_equal(result[name], expected[name])


def test_memory_stays_bounded_while_separating_into_files(tmp_path):
    """Peak allocation is a few chunks, not the length of the soundtrack."""
    length, chunk = 400_000, 4_000
    audio = noise(2, length)
    out = disk_stems(tmp_path, STEMS, audio.shape)
    tracemalloc.start()
    try:
        separate(Fake(chunk), audio, overlap=2, out=out)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 0.05 * 3 * audio.nbytes  # 3 stems in memory would be 3 * audio.nbytes
    np.testing.assert_allclose(out["speech"], audio, atol=1e-5)


def test_out_must_match_the_audio():
    audio = noise(2, 100)
    with pytest.raises(ValueError, match="out"):
        separate(Fake(), audio, out={"speech": np.zeros((2, 100), np.float32)})
    with pytest.raises(ValueError, match="out"):
        separate(Fake(), audio, out={n: np.zeros((2, 99), np.float32) for n in STEMS})
    with pytest.raises(ValueError, match="out"):
        separate(Fake(), audio, out={n: np.zeros((2, 100)) for n in STEMS})  # float64


def test_progress_reports_every_chunk():
    seen = []
    fake = Fake()
    separate(fake, noise(2, 4321), overlap=2, progress=lambda done, total: seen.append((done, total)))
    assert len(seen) == fake.calls
    assert seen[-1][0] == seen[-1][1]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        separate(Fake(), np.zeros(100, np.float32))
    with pytest.raises(ValueError):
        separate(Fake(), np.zeros((2, 100), np.float32), overlap=0)


def test_wrong_backend_shape_is_reported():
    class Broken(Fake):
        def separate_chunk(self, chunk):
            return chunk

    with pytest.raises(RuntimeError, match="shape"):
        separate(Broken(), noise(2, 3000))


def test_checkpoint_prefix_is_stripped():
    t = torch.zeros(1)
    assert match_keys({"model.a": t, "model.b": t, "other": t}, {"a", "b"}).keys() == {"a", "b"}
    with pytest.raises(RuntimeError, match="missing"):
        match_keys({"x": t}, {"a"})


def test_missing_weights_say_where_to_get_them(tmp_path):
    from musicremover.separate.bandit import BanditV2

    with pytest.raises(FileNotFoundError, match="zenodo"):
        BanditV2(tmp_path / "nope.ckpt")
