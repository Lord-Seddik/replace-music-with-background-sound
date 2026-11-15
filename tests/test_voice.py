import numpy as np
import pytest

from musicremover.separate import separate
from musicremover.separate.voice import rescue_gain, rescue_voices

RATE = 48000


def tone(freq: float, seconds: float, level: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    x = (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x])


def test_voice_alone_is_rescued_and_song_is_not():
    voice = tone(300, 2)
    # first second: voice alone (a laugh); second second: voice over instruments (a song)
    instruments = np.zeros_like(voice)
    instruments[:, RATE:] = tone(1000, 1, level=0.08)
    gain = rescue_gain(voice, instruments, RATE)
    assert gain.shape == (2 * RATE,)
    assert gain[int(0.2 * RATE):int(0.8 * RATE)].min() == pytest.approx(1.0)
    assert gain[int(1.2 * RATE):].max() == pytest.approx(0.0)


def test_gain_changes_smoothly():
    voice = tone(300, 2)
    instruments = np.zeros_like(voice)
    instruments[:, RATE:] = tone(1000, 1, level=0.08)
    gain = rescue_gain(voice, instruments, RATE)
    assert np.abs(np.diff(gain)).max() < 1 / (0.1 * RATE) * 1.01  # at most a 0.1 s ramp


def test_partial_rescue_between_thresholds():
    voice = tone(300, 1)
    instruments = voice * np.float32(10 ** (-25 / 20))  # voice 25 dB above instruments
    gain = rescue_gain(voice, instruments, RATE)
    assert gain[RATE // 4:3 * RATE // 4].mean() == pytest.approx(0.5, abs=0.02)


def test_silence_rescues_nothing():
    z = np.zeros((2, RATE), np.float32)
    assert not rescue_gain(z, z, RATE).any()


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        rescue_gain(np.zeros((2, 10), np.float32), np.zeros((2, 11), np.float32))


class FakeVoice:
    """Calls everything in the music stem 'vocals' (a laugh with no instruments)."""

    sample_rate = RATE
    chunk_size = 6000

    def separate_chunk(self, chunk):
        return np.stack([chunk, np.zeros_like(chunk)])


def test_rescue_moves_voice_from_music_to_speech_and_keeps_the_sum():
    rng = np.random.default_rng(1)
    stems = {k: rng.standard_normal((2, 20000)).astype(np.float32) * 0.1 for k in ("speech", "music", "effects")}
    out = rescue_voices(stems, FakeVoice())
    total_before = sum(stems.values())
    total_after = sum(out.values())
    np.testing.assert_allclose(total_after, total_before, atol=1e-5)
    np.testing.assert_allclose(out["speech"], stems["speech"] + stems["music"], atol=1e-4)
    np.testing.assert_allclose(out["music"], 0, atol=1e-4)
    np.testing.assert_array_equal(out["effects"], stems["effects"])


@pytest.mark.parametrize("overlap", [1.1, 1.5])
def test_fractional_overlap_is_seamless(overlap):
    audio = np.random.default_rng(0).standard_normal((2, 23456)).astype(np.float32)
    parts = separate(FakeVoice(), audio, overlap=overlap, names=("vocals", "instruments"))
    np.testing.assert_allclose(parts["vocals"], audio, atol=1e-5)
    np.testing.assert_allclose(parts["instruments"], 0, atol=1e-6)
