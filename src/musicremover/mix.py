"""Combine separated stems into the final soundtrack (pure numpy, no I/O)."""

from __future__ import annotations

import math
import re

import numpy as np

_REQUIRED = ("speech", "music", "effects")
_LOOKAHEAD_S = 0.005
_RELEASE_S = 0.100
_CHUNK = 1 << 22


def _validate(stems: dict[str, np.ndarray]) -> None:
    missing = [k for k in _REQUIRED if k not in stems]
    if missing:
        raise ValueError(f"missing stem(s): {', '.join(missing)}")
    shape = None
    for k in _REQUIRED:
        a = np.asarray(stems[k])
        if a.ndim != 2:
            raise ValueError(f"stem {k!r} must be 2-D (channels, samples), got {a.ndim}-D")
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise ValueError(f"stem shapes differ: {k!r} is {a.shape}, expected {shape}")


def _shift_left(x: np.ndarray, k: int) -> np.ndarray:
    """out[j] = x[j + k], zero padded at the end."""
    out = np.zeros_like(x)
    if k < len(x):
        out[: len(x) - k] = x[k:]
    return out


def _forward_window_max(x: np.ndarray, length: int) -> np.ndarray:
    """out[j] = max(x[j : j + length]) (x is non-negative, zero padded)."""
    r = x
    w = 1
    while w * 2 <= length:
        r = np.maximum(r, _shift_left(r, w))
        w *= 2
    if w < length:
        r = np.maximum(r, _shift_left(r, length - w))
    return r


def _decay_envelope(d: np.ndarray, decay: float) -> np.ndarray:
    """e[i] = max(d[i], decay * e[i-1]), vectorised in blocks."""
    n = len(d)
    out = np.empty(n)
    block = max(1, int(np.log(1e12) / -np.log(decay)))  # keep decay**-k in range
    carry = 0.0
    for s in range(0, n, block):
        seg = d[s : s + block]
        k = np.arange(len(seg), dtype=np.float64)
        env = np.maximum.accumulate(seg * decay ** -k) * decay**k
        env = np.maximum(env, carry * decay ** (k + 1))
        out[s : s + len(seg)] = env
        carry = env[-1]
    return out


def _limit_region(x: np.ndarray, ceiling: float, la: int, decay: float) -> None:
    """Apply the limiter in place to a (channels, samples) float32 slice."""
    peak = np.abs(x).max(axis=0).astype(np.float64)
    over = peak > ceiling
    deficit = np.zeros(len(peak))
    deficit[over] = 1.0 - ceiling / peak[over]
    # Look-ahead: reduction must already be in place `la` samples before a peak.
    deficit = _forward_window_max(deficit, la)
    deficit = _decay_envelope(deficit, decay)
    # Moving average over the past `la` samples gives a smooth attack ramp that
    # still reaches the full reduction when the peak arrives.
    # At the very start of the file there is no earlier audio to ramp in, so the
    # edge value is held (the gain simply starts already reduced).
    c = np.concatenate(([0.0], np.cumsum(np.concatenate((np.full(la, deficit[0]), deficit)))))
    avg = (c[la + 1 :] - c[1 : len(deficit) + 1]) / la
    gain = (1.0 - avg).astype(np.float32)
    x *= gain[None, :]
    # Final guarantee, linked across channels.
    peak = np.abs(x).max(axis=0)
    bad = peak > ceiling
    if bad.any():
        x[:, bad] *= np.float32(ceiling) / peak[bad] * np.float32(0.999999)
        np.clip(x, -ceiling, ceiling, out=x)


def _limit(x: np.ndarray, ceiling: float, sample_rate: int) -> np.ndarray:
    n = x.shape[1]
    hits = []
    for a in range(0, n, _CHUNK):
        p = np.abs(x[:, a : a + _CHUNK]).max(axis=0)
        h = np.flatnonzero(p > ceiling)
        if len(h):
            hits.append(h + a)
    if not hits:
        return x
    hits = np.concatenate(hits)
    la = max(1, int(round(_LOOKAHEAD_S * sample_rate)))
    rel = max(1, int(round(_RELEASE_S * sample_rate)))
    decay = float(np.exp(-1.0 / rel))
    tail = 7 * rel + 2 * la
    head = 2 * la
    breaks = np.flatnonzero(np.diff(hits) > tail + head) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(hits)])) - 1
    for si, ei in zip(starts, ends):
        s = max(0, int(hits[si]) - head)
        e = min(n, int(hits[ei]) + tail)
        _limit_region(x[:, s:e], ceiling, la, decay)
    return x


def mix(
    stems: dict[str, np.ndarray],
    music_gain_db: float | None = None,
    ceiling: float = 0.99,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Return speech + effects + music * gain, limited so it never exceeds ceiling.

    ``music_gain_db=None`` removes the music; a negative number keeps it that
    many dB quieter. Speech and effects are never rescaled.
    """
    _validate(stems)
    if music_gain_db is not None and music_gain_db > 0:
        raise ValueError("music_gain_db must be <= 0 (music is never made louder)")
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")

    out = np.asarray(stems["speech"], dtype=np.float32) + np.asarray(
        stems["effects"], dtype=np.float32
    )
    if music_gain_db is not None:
        g = np.float32(10.0 ** (music_gain_db / 20.0))
        out += np.asarray(stems["music"], dtype=np.float32) * g
    return _limit(out, ceiling, sample_rate)


_INT = re.compile(r"\d+")
_SEC = re.compile(r"\d+(\.\d+)?")


def parse_time(text: str) -> float:
    """Parse ``SS``, ``M:SS`` or ``H:MM:SS`` (seconds may have decimals) to seconds."""
    t = text.strip()
    if not t:
        raise ValueError("empty time")
    parts = t.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time {text!r}: use SS, M:SS or H:MM:SS")
    if not _SEC.fullmatch(parts[-1]) or not all(_INT.fullmatch(p) for p in parts[:-1]):
        raise ValueError(f"bad time {text!r}: use SS, M:SS or H:MM:SS (no negatives)")
    if len(parts) > 1:
        if float(parts[-1]) >= 60:
            raise ValueError(f"bad time {text!r}: seconds must be below 60")
        if len(parts) == 3 and int(parts[1]) >= 60:
            raise ValueError(f"bad time {text!r}: minutes must be below 60")
    total = 0
    for p in parts[:-1]:
        total = total * 60 + int(p)
    return total * 60 + float(parts[-1])


def parse_range(text: str) -> tuple[float, float]:
    """Parse ``START-END`` (END may be ``end``) into seconds; end is inf for ``end``."""
    if text.count("-") != 1:
        raise ValueError(f"bad range {text!r}: use START-END, e.g. 0:00-1:30 or 22:10-end")
    a, b = text.split("-")
    start = parse_time(a)
    b = b.strip()
    end = math.inf if b.lower() == "end" else parse_time(b)
    if end <= start:
        raise ValueError(f"bad range {text!r}: end must be after start")
    return start, end


def keep_ranges(
    mixed: np.ndarray,
    original: np.ndarray,
    ranges: list[tuple[float, float]],
    sample_rate: int = 48000,
    fade: float = 0.05,
) -> np.ndarray:
    """Return ``mixed`` with the given time ranges replaced by ``original``.

    Each range edge gets a linear crossfade of ``fade`` seconds lying inside the
    range. Edges at the very start or end of the signal are not faded.
    """
    mixed = np.asarray(mixed)
    original = np.asarray(original)
    if mixed.shape != original.shape or mixed.ndim != 2:
        raise ValueError("mixed and original must be 2-D arrays of the same shape")
    n = mixed.shape[1]
    out = mixed.copy()

    spans: list[list[int]] = []
    for s, e in sorted(ranges):
        a = n if s * sample_rate >= n else int(round(s * sample_rate))
        b = n if math.isinf(e) or e * sample_rate >= n else int(round(e * sample_rate))
        if b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])

    fade_n = max(0, int(round(fade * sample_rate)))
    for a, b in spans:
        f = min(fade_n, (b - a) // 2)
        out[:, a:b] = original[:, a:b]
        if f == 0:
            continue
        w = (np.arange(1, f + 1, dtype=np.float32) / np.float32(f + 1))[None, :]
        if a > 0:
            out[:, a : a + f] = mixed[:, a : a + f] * (1 - w) + original[:, a : a + f] * w
        if b < n:
            wr = w[:, ::-1]
            out[:, b - f : b] = mixed[:, b - f : b] * (1 - wr) + original[:, b - f : b] * wr
    return out
