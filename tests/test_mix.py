import time

import numpy as np
import pytest

from musicremover.mix import mix

SR = 48000


def _sine(freq, amp, n=SR, ch=2):
    t = np.arange(n) / SR
    s = amp * np.sin(2 * np.pi * freq * t)
    return np.tile(s, (ch, 1)).astype(np.float32)


def _stems(speech, music, effects):
    return {"speech": speech, "music": music, "effects": effects}


def test_default_removes_music_exactly():
    sp, mu, fx = _sine(220, 0.2), _sine(440, 0.3), _sine(1000, 0.1)
    out = mix(_stems(sp, mu, fx))
    assert np.array_equal(out, sp + fx)


def test_music_gain_minus_20_adds_tenth():
    sp, mu, fx = _sine(220, 0.2), _sine(440, 0.3), _sine(1000, 0.1)
    out = mix(_stems(sp, mu, fx), music_gain_db=-20)
    np.testing.assert_allclose(out, sp + fx + 0.1 * mu, atol=1e-6)


def test_positive_gain_rejected():
    s = _sine(220, 0.1)
    with pytest.raises(ValueError):
        mix(_stems(s, s, s), music_gain_db=3)


def test_missing_stem_rejected():
    s = _sine(220, 0.1)
    with pytest.raises(ValueError):
        mix({"speech": s, "music": s})


def test_shape_mismatch_and_ndim_rejected():
    s = _sine(220, 0.1)
    with pytest.raises(ValueError):
        mix(_stems(s, s[:, :100], s))
    with pytest.raises(ValueError):
        mix(_stems(s[0], s[0], s[0]))


def test_quiet_signal_unchanged_bit_for_bit():
    rng = np.random.default_rng(0)
    sp = (rng.standard_normal((2, SR)) * 0.1).astype(np.float32)
    mu = (rng.standard_normal((2, SR)) * 0.1).astype(np.float32)
    fx = (rng.standard_normal((2, SR)) * 0.1).astype(np.float32)
    out = mix(_stems(sp, mu, fx), music_gain_db=-6)
    expected = sp + fx
    expected += mu * np.float32(10 ** (-6 / 20))
    assert np.abs(expected).max() < 0.99
    assert np.array_equal(out, expected)


def _loud():
    n = 2 * SR
    return _sine(220, 0.65, n), _sine(440, 0.65, n), _sine(330, 0.65, n)


def _gain_curve(out, raw):
    """Gain per sample, and a mask of samples where raw is big enough to measure it."""
    m = np.abs(raw) > 0.05
    gain = np.ones(len(raw))
    gain[m] = out[m] / raw[m]
    return gain, m


def test_loud_signal_never_exceeds_ceiling_and_gain_is_smooth():
    sp, mu, fx = _loud()
    out = mix(_stems(sp, mu, fx), music_gain_db=0.0)
    raw = sp + fx + mu
    assert np.abs(raw).max() > 1.6
    assert np.abs(out).max() <= 0.99
    gain, m = _gain_curve(out[0], raw[0])
    assert gain[m].min() < 0.7  # limiter actually worked
    both = m[1:] & m[:-1]
    assert np.abs(np.diff(gain))[both].max() < 0.01


def test_channels_get_same_gain():
    sp, mu, fx = _loud()
    sp[1] *= 0.5  # different level per channel, same limiter gain
    out = mix(_stems(sp, mu, fx), music_gain_db=0.0)
    raw = sp + fx + mu
    m = (np.abs(raw[0]) > 0.3) & (np.abs(raw[1]) > 0.3)
    np.testing.assert_allclose(
        out[0][m] / raw[0][m], out[1][m] / raw[1][m], rtol=1e-4, atol=1e-5
    )


def test_lookahead_gain_down_before_peak():
    sp = np.full((2, SR), 0.5, dtype=np.float32)
    sp[:, 24000] = 1.8  # single spike
    z = np.zeros_like(sp)
    out = mix(_stems(sp, z, z))
    assert out[0, 24000] <= 0.99
    assert out[0, 24000 - 100] < 0.5  # already reduced before the spike
    assert out[0, 24000 + 4800] > 0.4  # recovering afterwards


def test_dtype_and_shape_preserved():
    sp, mu, fx = _loud()
    out = mix(_stems(sp, mu, fx), music_gain_db=-3)
    assert out.dtype == np.float32 and out.shape == sp.shape
    out = mix(_stems(sp.astype(np.float64), mu, fx))
    assert out.dtype == np.float32


def test_speed_24_minute_stereo_with_loud_bursts():
    n = 24 * 60 * SR
    rng = np.random.default_rng(1)
    sp = rng.standard_normal((2, n), dtype=np.float32) * np.float32(0.1)
    fx = np.zeros_like(sp)
    mu = np.zeros_like(sp)
    for pos in (n // 5, n // 2, 4 * n // 5):
        sp[:, pos : pos + SR // 2] += 1.5
    t0 = time.perf_counter()
    out = mix(_stems(sp, mu, fx), music_gain_db=-10)
    elapsed = time.perf_counter() - t0
    print(f"24-minute limiter speed test: {elapsed:.2f}s")
    assert np.abs(out).max() <= 0.99
    assert elapsed < 30


# ---- parse_time / parse_range / keep_ranges ----
from musicremover.mix import keep_ranges, parse_range, parse_time  # noqa: E402


@pytest.mark.parametrize("text,want", [
    ("5", 5.0), ("0", 0.0), ("90", 90.0), ("1:30", 90.0), ("1:30.5", 90.5),
    ("0:05", 5.0), ("1:02:03", 3723.0), ("1:02:03.25", 3723.25), (" 7 ", 7.0),
])
def test_parse_time_valid(text, want):
    assert parse_time(text) == pytest.approx(want)


@pytest.mark.parametrize("text", [
    "", " ", "-5", "abc", "1:60", "1:75", "1:60:00", "1:2:3:4", ":30", "1:", "1.5:30",
    "1:-5", "1e3", "1,5",
])
def test_parse_time_invalid(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_parse_range():
    assert parse_range("0:00-1:30") == (0.0, 90.0)
    assert parse_range("22:10-end") == (1330.0, float("inf"))
    assert parse_range("5-6.5") == (5.0, 6.5)


@pytest.mark.parametrize("text", ["5-5", "10-5", "banana", "1-2-3", "5", "-5", "5-", "1:99-2:00"])
def test_parse_range_invalid(text):
    with pytest.raises(ValueError):
        parse_range(text)


def _pair(n=4 * SR, ch=2):
    t = np.arange(n) / SR
    mixed = np.tile((0.3 * np.sin(2 * np.pi * 50 * t)).astype(np.float32), (ch, 1))
    orig = np.tile((0.6 * np.sin(2 * np.pi * 70 * t + 1)).astype(np.float32), (ch, 1))
    return mixed, orig


FADE = int(0.05 * SR)


def test_keep_inside_and_outside_exact():
    m, o = _pair()
    out = keep_ranges(m, o, [(1.0, 2.0)], SR)
    assert out.dtype == np.float32 and out.shape == m.shape
    assert np.array_equal(out[:, SR + FADE : 2 * SR - FADE], o[:, SR + FADE : 2 * SR - FADE])
    assert np.array_equal(out[:, :SR], m[:, :SR])
    assert np.array_equal(out[:, 2 * SR :], m[:, 2 * SR :])


def test_keep_fade_between_and_continuous():
    m, o = _pair()
    out = keep_ranges(m, o, [(1.0, 2.0)], SR)
    for sl in (slice(SR, SR + FADE), slice(2 * SR - FADE, 2 * SR)):
        lo = np.minimum(m[:, sl], o[:, sl]) - 1e-6
        hi = np.maximum(m[:, sl], o[:, sl]) + 1e-6
        assert np.all(out[:, sl] >= lo) and np.all(out[:, sl] <= hi)
    step = max(np.abs(np.diff(m)).max(), np.abs(np.diff(o)).max())
    assert np.abs(np.diff(out)).max() <= 2 * step


def test_keep_overlapping_ranges_merged():
    m, o = _pair()
    a = keep_ranges(m, o, [(1.0, 2.0), (1.5, 2.5)], SR)
    b = keep_ranges(m, o, [(1.0, 2.5)], SR)
    c = keep_ranges(m, o, [(1.0, 1.5), (1.5, 2.5)], SR)
    assert np.array_equal(a, b) and np.array_equal(c, b)
    mid = slice(int(1.9 * SR), int(2.1 * SR))
    assert np.array_equal(b[:, mid], o[:, mid])


def test_keep_unsorted_ranges():
    m, o = _pair()
    assert np.array_equal(keep_ranges(m, o, [(3.0, 3.5), (1.0, 1.5)], SR),
                          keep_ranges(m, o, [(1.0, 1.5), (3.0, 3.5)], SR))


def test_keep_clamped_and_end():
    m, o = _pair()
    out = keep_ranges(m, o, [(3.0, 100.0)], SR)
    assert np.array_equal(out[:, 3 * SR + FADE :], o[:, 3 * SR + FADE :])
    assert np.array_equal(out[:, : 3 * SR], m[:, : 3 * SR])
    inf = keep_ranges(m, o, [(3.0, float("inf"))], SR)
    assert np.array_equal(inf, out)
    assert out.shape == m.shape


def test_keep_from_zero_not_faded():
    m, o = _pair()
    out = keep_ranges(m, o, [(0.0, 1.0)], SR)
    assert np.array_equal(out[:, : SR - FADE], o[:, : SR - FADE])
    assert np.array_equal(out[:, SR:], m[:, SR:])


def test_keep_range_past_end_ignored():
    m, o = _pair()
    assert np.array_equal(keep_ranges(m, o, [(10.0, 20.0), (4.0, 5.0)], SR), m)


def test_keep_short_range():
    m, o = _pair()
    a, b = SR, SR + int(0.04 * SR)  # shorter than two fades
    out = keep_ranges(m, o, [(a / SR, b / SR)], SR)
    assert np.array_equal(out[:, :a], m[:, :a])
    assert np.array_equal(out[:, b:], m[:, b:])
    assert not np.array_equal(out[:, a:b], m[:, a:b])
    seg = out[:, a:b]
    lo = np.minimum(m[:, a:b], o[:, a:b]) - 1e-6
    hi = np.maximum(m[:, a:b], o[:, a:b]) + 1e-6
    assert np.all(seg >= lo) and np.all(seg <= hi)


def test_keep_inputs_unchanged():
    m, o = _pair()
    m0, o0 = m.copy(), o.copy()
    out = keep_ranges(m, o, [(1.0, 2.0)], SR)
    assert np.array_equal(m, m0) and np.array_equal(o, o0)
    assert out is not m and out is not o
    assert keep_ranges(m, o, [], SR) is not m
