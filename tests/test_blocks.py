"""Block-wise mix, keep, voice rescue and fill give the same audio as the whole-array
versions (frozen copies of the pre-T20 code below), and need little memory.

Tolerances: keep, the fill and `excerpt` are bit-identical; the voice rescue agrees to
atol 1e-6 (frame powers are summed in different groups); the limiter carries its
envelope from block to block and sums it in a different order, so it agrees to atol 2e-6.
"""
import math
import tracemalloc

import numpy as np
import pytest

from musicremover import library, mix as mixmod
from musicremover.library import Library, Placement, add_fill, render_fill
from musicremover.mix import keep_blocks, keep_ranges, mix, mix_blocks
from musicremover.separate import STEMS, disk_stems, separate, voice
from musicremover.separate.voice import rescue_voices

RATE = 8000


# ---- frozen whole-array references ------------------------------------------------------

def ref_limit_region(x, ceiling, la, decay):
    peak = np.abs(x).max(axis=0).astype(np.float64)
    over = peak > ceiling
    deficit = np.zeros(len(peak))
    deficit[over] = 1.0 - ceiling / peak[over]
    deficit = mixmod._forward_window_max(deficit, la)
    deficit = mixmod._decay_envelope(deficit, decay)
    c = np.concatenate(([0.0], np.cumsum(np.concatenate((np.full(la, deficit[0]), deficit)))))
    avg = (c[la + 1:] - c[1:len(deficit) + 1]) / la
    x *= (1.0 - avg).astype(np.float32)[None, :]
    peak = np.abs(x).max(axis=0)
    bad = peak > ceiling
    if bad.any():
        x[:, bad] *= np.float32(ceiling) / peak[bad] * np.float32(0.999999)
        np.clip(x, -ceiling, ceiling, out=x)


def ref_mix(stems, g_db, ceiling, rate):
    x = stems["speech"].astype(np.float32) + stems["effects"].astype(np.float32)
    if g_db is not None:
        x += stems["music"].astype(np.float32) * np.float32(10.0 ** (g_db / 20.0))
    n = x.shape[1]
    hits = np.flatnonzero(np.abs(x).max(axis=0) > ceiling)
    if not len(hits):
        return x
    la = max(1, int(round(0.005 * rate)))
    rel = max(1, int(round(0.1 * rate)))
    decay = float(np.exp(-1.0 / rel))
    tail, head = 7 * rel + 2 * la, 2 * la
    breaks = np.flatnonzero(np.diff(hits) > tail + head) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(hits)])) - 1
    for si, ei in zip(starts, ends):
        ref_limit_region(x[:, max(0, int(hits[si]) - head):min(n, int(hits[ei]) + tail)], ceiling, la, decay)
    return x


def ref_keep(mixed, original, ranges, rate, fade=0.05):
    n = mixed.shape[1]
    out = mixed.copy()
    spans = []
    for s, e in sorted(ranges):
        a = n if s * rate >= n else int(round(s * rate))
        b = n if math.isinf(e) or e * rate >= n else int(round(e * rate))
        if b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    fade_n = max(0, int(round(fade * rate)))
    for a, b in spans:
        f = min(fade_n, (b - a) // 2)
        out[:, a:b] = original[:, a:b]
        if f == 0:
            continue
        w = (np.arange(1, f + 1, dtype=np.float32) / np.float32(f + 1))[None, :]
        if a > 0:
            out[:, a:a + f] = mixed[:, a:a + f] * (1 - w) + original[:, a:a + f] * w
        if b < n:
            wr = w[:, ::-1]
            out[:, b - f:b] = mixed[:, b - f:b] * (1 - wr) + original[:, b - f:b] * wr
    return out


def ref_rescue_gain(vocals, instruments, rate, frame_s=0.1, low_db=20.0, high_db=30.0):
    length = vocals.shape[1]
    frame = max(1, int(frame_s * rate))
    count = -(-length // frame)

    def power(x):
        y = np.zeros((x.shape[0], count * frame), np.float32)
        y[:, :length] = x
        return (y.astype(np.float64) ** 2).reshape(x.shape[0], count, frame).mean(axis=(0, 2))

    ratio = 10 * np.log10((power(vocals) + 1e-12) / (power(instruments) + 1e-12))
    gains = np.clip((ratio - low_db) / (high_db - low_db), 0.0, 1.0)
    centres = (np.arange(count) + 0.5) * frame
    return np.interp(np.arange(length), centres, gains).astype(np.float32), gains


def ref_rescue(stems, sep):
    parts = separate(sep, stems["music"], overlap=1.1, names=voice.VOICE_NAMES)
    gain, gains = ref_rescue_gain(parts["vocals"], parts["instruments"], sep.sample_rate)
    rescued = parts["vocals"] * gain
    out = {"speech": stems["speech"] + rescued, "music": stems["music"] - rescued,
           "effects": stems["effects"]}
    return out, gains


def ref_excerpt(audio, length, rng, loop_fade):
    m = audio.shape[1]
    if m >= length:
        off = int(rng.integers(0, m - length + 1))
        return audio[:, off:off + length].copy()
    xf = max(1, min(loop_fade, m // 4))
    out = np.zeros((audio.shape[0], length), np.float32)
    pos, first = 0, True
    while pos < length:
        s = 0 if first else int(rng.integers(0, m // 2 + 1))
        chunk = audio[:, s:].copy()
        if not first:
            chunk[:, :xf] *= library._ramp(xf)
        chunk[:, -xf:] *= library._ramp(xf)[::-1]
        n = min(chunk.shape[1], length - pos)
        out[:, pos:pos + n] += chunk[:, :n]
        pos += chunk.shape[1] - xf
        first = False
    return out


def ref_envelope(length, fade_in, fade_out):
    env = np.ones(length, np.float32)
    fade_in, fade_out = min(fade_in, length // 2), min(fade_out, length // 2)
    if fade_in:
        env[:fade_in] *= library._ramp(fade_in)
    if fade_out:
        env[length - fade_out:] *= library._ramp(fade_out)[::-1]
    return env


def ref_render(placements, lib, channels, length, rate, read, level_offset_db=0.0):
    out = np.zeros((channels, length), np.float32)
    rng = np.random.default_rng(0)
    cache = {}
    edge, scene_fade = int(1.0 * rate), int(0.5 * rate)
    half = 0.25
    for k, pl in enumerate(placements):
        before = k > 0 and abs(placements[k - 1].end - pl.start) < 1e-6
        after = k + 1 < len(placements) and abs(placements[k + 1].start - pl.end) < 1e-6
        a = max(0, int(round((pl.start - (half if before else 0)) * rate)))
        b = min(length, int(round((pl.end + (half if after else 0)) * rate)))
        if b <= a:
            continue
        files = lib.sounds[pl.scene]
        path = files[int(rng.integers(len(files)))]
        if path not in cache:
            cache[path] = read(path, rate, channels)
        audio, level = cache[path]
        seg = ref_excerpt(audio, b - a, rng, int(2.0 * rate))
        gain = np.float32(10 ** ((pl.level_db + level_offset_db - level) / 20))
        env = ref_envelope(b - a, scene_fade if before else edge, scene_fade if after else edge)
        out[:, a:b] += seg * gain * env
    return out


# ---- helpers ---------------------------------------------------------------------------

def noise(channels, n, level=0.3, seed=0):
    return (np.random.default_rng(seed).standard_normal((channels, n)) * level).astype(np.float32)


def collect(blocks, shape):
    out = np.full(shape, np.nan, np.float32)
    pos = 0
    for a, y in blocks:
        assert a == pos, "blocks must follow each other"
        out[:, a:a + y.shape[1]] = y
        pos += y.shape[1]
    assert pos == shape[1]
    return out


def loud_stems(n=60000, seed=1):
    """Speech with bursts above the ceiling: one right at the start, several close
    together (the limiter's decay runs on between them), a long clipped stretch crossing
    many blocks, bursts a little less and a little more than one limiter stretch apart,
    and one at the very end."""
    rng = np.random.default_rng(seed)
    speech = (rng.standard_normal((2, n)) * 0.05).astype(np.float32)
    gap = 7 * 800 + 4 * 40  # RATE 8000: release 800, look-ahead 40
    for pos, length in [(3, 5), (1000, 3), (2300, 2), (9000, 12000), (9000 + 12000 + gap - 1, 4),
                        (9000 + 12000 + gap - 1 + 4 + gap + 1, 4), (n - 3, 3)]:
        speech[:, pos:pos + length] += 1.4
    speech[1, 30001] -= 2.0
    return {"speech": speech, "music": noise(2, n, 0.2, seed + 1), "effects": noise(2, n, 0.02, seed + 2)}


# ---- limiter / mix ---------------------------------------------------------------------

@pytest.mark.parametrize("block", [1, 200, 777, 4096, 100000])
@pytest.mark.parametrize("music_gain_db", [None, -6.0])
def test_mix_blocks_match_whole_array_limiter(block, music_gain_db):
    stems = loud_stems()
    want = ref_mix(stems, music_gain_db, 0.99, RATE)
    got = collect(mix_blocks(stems, music_gain_db, 0.99, RATE, block), want.shape)
    assert np.abs(want).max() <= 0.99
    np.testing.assert_allclose(got, want, atol=2e-6, rtol=0)
    np.testing.assert_allclose(mix(stems, music_gain_db, 0.99, RATE, block), want, atol=2e-6, rtol=0)


def test_mix_blocks_untouched_where_nothing_is_loud():
    stems = {k: noise(2, 30000, 0.05, i) for i, k in enumerate(("speech", "music", "effects"))}
    got = collect(mix_blocks(stems, -3.0, 0.99, RATE, 1000), (2, 30000))
    np.testing.assert_array_equal(got, ref_mix(stems, -3.0, 0.99, RATE))


def test_mix_blocks_read_memory_mapped_stems(tmp_path):
    stems = loud_stems()
    mapped = disk_stems(tmp_path, STEMS, stems["speech"].shape)
    for k in STEMS:
        mapped[k][:] = stems[k]
    got = collect(mix_blocks(mapped, None, 0.99, RATE, 3000), stems["speech"].shape)
    np.testing.assert_allclose(got, ref_mix(stems, None, 0.99, RATE), atol=2e-6, rtol=0)
    del mapped


# ---- --keep ----------------------------------------------------------------------------

KEEP_RANGES = [
    [(1.0, 2.0)],                                   # crossfade edges inside blocks
    [(0.0, 0.6), (3.0, math.inf)],                  # at the very start and to the end (no fade there)
    [(1.0, 1.02), (2.0, 2.5), (2.4, 3.0)],          # shorter than the fade, and overlapping
    [(0.0, math.inf)],
    [(5.5, 6.0)],                                   # past the end
]


@pytest.mark.parametrize("ranges", KEEP_RANGES)
@pytest.mark.parametrize("block", [7, 333, 1000, 5000])
def test_keep_blocks_equal_whole_array(ranges, block):
    n = 4 * RATE  # the fade is 0.05 s = 400 samples, so edges fall inside and between blocks
    mixed, original = noise(2, n, seed=3), noise(2, n, seed=4)
    want = ref_keep(mixed, original, ranges, RATE)
    pieces = [(a, mixed[:, a:a + block]) for a in range(0, n, block)]
    got = collect(keep_blocks(pieces, original, ranges, RATE), want.shape)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(keep_ranges(mixed, original, ranges, RATE), want)


def test_keep_blocks_do_not_change_the_blocks_they_are_given():
    mixed, original = noise(2, 2000), noise(2, 2000, seed=9)
    block = mixed.copy()
    list(keep_blocks([(0, block)], original, [(0.0, 0.2)], RATE))
    np.testing.assert_array_equal(block, mixed)


# ---- voice rescue ----------------------------------------------------------------------

class SplitVoice:
    """'vocals' is the chunk; 'instruments' a copy whose level changes from chunk to
    chunk, so the rescue gain takes values from 0 to 1."""

    sample_rate = RATE
    chunk_size = 2500

    def __init__(self):
        self.calls = 0

    def separate_chunk(self, chunk):
        self.calls += 1
        db = (-5, -22, -26, -45)[self.calls % 4]
        return np.stack([chunk, chunk * np.float32(10 ** (db / 20))])


def stems_of(n, seed=0):
    return {k: noise(2, n, 0.1, seed + i) for i, k in enumerate(STEMS)}


@pytest.mark.parametrize("block, frames", [(1, 1), (700, 3), (4096, 2), (1 << 20, 1 << 8)])
def test_rescue_voices_blockwise_equals_whole_array(monkeypatch, block, frames):
    monkeypatch.setattr(voice, "_BLOCK", block)
    monkeypatch.setattr(voice, "_BLOCK_FRAMES", frames)
    stems = stems_of(30001)
    want, gains = ref_rescue(stems, SplitVoice())
    assert ((gains > 0) & (gains < 1)).any() and (gains == 0).any() and (gains == 1).any()  # all kinds of gain occur
    got = rescue_voices(stems, SplitVoice())
    for k in STEMS:
        np.testing.assert_allclose(got[k], want[k], atol=1e-6, rtol=0)
    assert (got["speech"] != stems["speech"]).any() and (got["music"] != stems["music"]).any()


def test_rescue_voices_in_place_on_disk_with_scratch_files(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_BLOCK", 1234)
    monkeypatch.setattr(voice, "_BLOCK_FRAMES", 2)
    plain = stems_of(30001)
    want, _ = ref_rescue(plain, SplitVoice())
    mapped = disk_stems(tmp_path, STEMS, plain["speech"].shape)
    for k in STEMS:
        mapped[k][:] = plain[k]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    got = rescue_voices(mapped, SplitVoice(), out=mapped, scratch=scratch)
    assert got["speech"] is mapped["speech"]
    for k in STEMS:
        np.testing.assert_allclose(got[k], want[k], atol=1e-6, rtol=0)
    assert not list(scratch.iterdir())  # the voice/instrument split is cleaned up
    del got, mapped


def test_rescue_gain_blockwise_equals_whole_array(monkeypatch):
    monkeypatch.setattr(voice, "_BLOCK_FRAMES", 3)
    monkeypatch.setattr(voice, "_BLOCK", 999)
    v, i = noise(2, 30001, 0.1, 5), noise(2, 30001, 0.1, 6) * np.float32(0.05)
    i[:, 12000:] *= 20
    np.testing.assert_allclose(voice.rescue_gain(v, i, RATE), ref_rescue_gain(v, i, RATE)[0], atol=1e-6, rtol=0)


# ---- fill ------------------------------------------------------------------------------

def fake_read(path, rate, channels):
    seconds = 2.0 if path.name.startswith("short") else 40.0
    return noise(channels, int(seconds * rate), 0.1, seed=len(path.name)), -20.0


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path, {"park": (tmp_path / "short1.wav", tmp_path / "short22.wav"),
                              "beach": (tmp_path / "long.wav",)})


PLACEMENTS = [
    Placement(1.0, 9.0, "park", -45.0, 0),      # a 2 s recording looped over 8 s, 1 s edge fades
    Placement(9.0, 20.0, "beach", -42.0, 0),    # touching: 0.5 s crossfade, from a long recording
    Placement(20.0, 31.0, "park", -45.0, 0),    # crossfade again, looped
    Placement(40.0, 41.2, "beach", -50.0, 1),   # shorter than two edge fades
    Placement(45.0, 70.0, "park", -45.0, 2),    # to the end of the signal
]


@pytest.mark.parametrize("block", [1000, 3333, 8000, 1 << 20])
def test_render_fill_blockwise_equals_whole_array(lib, block):
    length = 60 * RATE
    want = ref_render(PLACEMENTS, lib, 2, length, RATE, fake_read, 3.0)
    got = render_fill(PLACEMENTS, lib, 2, length, RATE, read=fake_read, level_offset_db=3.0, block=block)
    np.testing.assert_array_equal(got, want)
    assert abs(got).max() > 0


def test_render_fill_into_a_file_and_add_it(lib, tmp_path):
    length = 60 * RATE
    want = ref_render(PLACEMENTS, lib, 2, length, RATE, fake_read)
    out = np.lib.format.open_memmap(tmp_path / "fill.npy", mode="w+", dtype=np.float32, shape=(2, length))
    got = render_fill(PLACEMENTS, lib, 2, length, RATE, read=fake_read, out=out, block=4001)
    assert got is out
    np.testing.assert_array_equal(out, want)
    effects = noise(2, length, 0.01)
    expected = effects + want
    add_fill(effects, out, block=5003)
    np.testing.assert_array_equal(effects, expected)
    del got, out


def test_excerpt_still_matches_the_old_one():
    for m, length in [(3000, 1000), (3000, 17000), (500, 4999), (40, 40), (40, 41)]:
        audio = noise(2, m, seed=m)
        a = library.excerpt(audio, length, np.random.default_rng(2), 400)
        b = ref_excerpt(audio, length, np.random.default_rng(2), 400)
        np.testing.assert_array_equal(a, b)


# ---- memory ----------------------------------------------------------------------------

def test_mix_keep_and_fill_memory_stays_bounded(tmp_path, lib):
    """Peak allocation is a few blocks, not the length of the signal."""
    n, block = 3_000_000, 50_000
    stems = disk_stems(tmp_path, STEMS, (2, n))
    for k in STEMS:
        for a in range(0, n, 500_000):
            stems[k][:, a:a + 500_000] = noise(2, min(500_000, n - a), 0.3, a + len(k))
    original = np.lib.format.open_memmap(tmp_path / "orig.npy", mode="w+", dtype=np.float32, shape=(2, n))
    original[:] = 0.1
    out = np.lib.format.open_memmap(tmp_path / "fill.npy", mode="w+", dtype=np.float32, shape=(2, n))
    placements = [Placement(1.0, 300.0, "park", -45.0, 0), Placement(300.0, 370.0, "beach", -45.0, 0)]
    tracemalloc.start()
    try:
        render_fill(placements, lib, 2, n, RATE, read=fake_read, out=out, block=block)
        add_fill(stems["effects"], out, block=block)
        total = 0
        for a, y in keep_blocks(mix_blocks(stems, -6.0, 0.99, RATE, block), original, [(10.0, 100.0)], RATE):
            total += y.shape[1]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert total == n
    # held: the two recordings and the noise made to fake them (some MB), nothing of signal length;
    # one stem is 24 MB and the whole-array version holds several
    assert peak < 0.5 * 2 * n * 4, peak
    del stems, original, out
