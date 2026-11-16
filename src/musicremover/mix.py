"""Combine separated stems into the final soundtrack (pure numpy, no I/O)."""

from __future__ import annotations

import math
import re

from collections.abc import Callable, Iterable, Iterator

import numpy as np

_REQUIRED = ("speech", "music", "effects")
_LOOKAHEAD_S = 0.005
_RELEASE_S = 0.100
_BLOCK = 1 << 20  # samples per step


def _validate(stems: dict[str, np.ndarray]) -> None:
    missing = [k for k in _REQUIRED if k not in stems]
    if missing:
        raise ValueError(f"missing stem(s): {', '.join(missing)}")
    shape = None
    for k in _REQUIRED:
        a = stems[k] if hasattr(stems[k], "shape") else np.asarray(stems[k])
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


def _decay_envelope(d: np.ndarray, decay: float, carry: float = 0.0) -> np.ndarray:
    """e[i] = max(d[i], decay * e[i-1]) with e[-1] = carry, vectorised in blocks."""
    n = len(d)
    out = np.empty(n)
    block = max(1, int(np.log(1e12) / -np.log(decay)))  # keep decay**-k in range
    for s in range(0, n, block):
        seg = d[s : s + block]
        k = np.arange(len(seg), dtype=np.float64)
        env = np.maximum.accumulate(seg * decay ** -k) * decay**k
        env = np.maximum(env, carry * decay ** (k + 1))
        out[s : s + len(seg)] = env
        carry = env[-1]
    return out


def _limit_stretch(read: Callable[[int, int], np.ndarray], start: int, stop: int, ceiling: float,
                   la: int, decay: float, block: int) -> Iterator[tuple[int, np.ndarray]]:
    """Limit samples [start, stop) of the signal `read(a, b)` returns, `block` samples at a time.

    Yields (position, limited block). The gain is the same as if the whole stretch
    were processed at once: the decay envelope carries over from block to block, the
    attack average keeps the last `la` envelope values, and each block looks `la - 1`
    samples into the next one before changing anything.
    """
    carry = 0.0
    hist: np.ndarray | None = None
    for a in range(start, stop, block):
        b = min(stop, a + block)
        x = read(a, min(stop, b + la - 1))
        peak = np.abs(x).max(axis=0).astype(np.float64)
        over = peak > ceiling
        deficit = np.zeros(len(peak))
        deficit[over] = 1.0 - ceiling / peak[over]
        # Look-ahead: reduction must already be in place `la` samples before a peak.
        deficit = _forward_window_max(deficit, la)[: b - a]
        env = _decay_envelope(deficit, decay, carry)
        carry = env[-1]
        # At the very start of the stretch there is no earlier audio to ramp in, so the
        # edge value is held (the gain simply starts already reduced).
        if hist is None:
            hist = np.full(la, env[0])
        # Moving average over the past `la` samples gives a smooth attack ramp that
        # still reaches the full reduction when the peak arrives.
        padded = np.concatenate((hist, env))
        c = np.concatenate(([0.0], np.cumsum(padded)))
        avg = (c[la + 1:] - c[1:len(env) + 1]) / la
        hist = padded[-la:]
        y = np.array(x[:, : b - a], dtype=np.float32)
        y *= (1.0 - avg).astype(np.float32)[None, :]
        # Final guarantee, linked across channels.
        peak = np.abs(y).max(axis=0)
        bad = peak > ceiling
        if bad.any():
            y[:, bad] *= np.float32(ceiling) / peak[bad] * np.float32(0.999999)
            np.clip(y, -ceiling, ceiling, out=y)
        yield a, y


def _find_stretches(read: Callable[[int, int], np.ndarray], n: int, ceiling: float, gap: int,
                    head: int, tail: int, block: int) -> list[tuple[int, int]]:
    """Stretches [start, stop) that need limiting: every sample above the ceiling,
    `head` samples before it and `tail` after it; hits closer than `gap` share one."""
    found: list[tuple[int, int]] = []
    first = last = None
    for a in range(0, n, block):
        hits = np.flatnonzero(np.abs(read(a, min(n, a + block))).max(axis=0) > ceiling)
        if not len(hits):
            continue
        hits = hits + a
        breaks = np.flatnonzero(np.diff(hits) > gap) + 1
        for lo, hi in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [len(hits)])) - 1):
            if last is not None and hits[lo] - last <= gap:
                last = int(hits[hi])
            else:
                if last is not None:
                    found.append((first, last))
                first, last = int(hits[lo]), int(hits[hi])
    if last is not None:
        found.append((first, last))
    return [(max(0, f - head), min(n, l + tail)) for f, l in found]


def limit_blocks(read: Callable[[int, int], np.ndarray], n: int, ceiling: float, sample_rate: int,
                 block: int = _BLOCK) -> Iterator[tuple[int, np.ndarray]]:
    """The limiter over the signal `read(a, b)` of `n` samples, as (position, block) pairs
    in order that together cover [0, n). Untouched stretches pass through unchanged."""
    la = max(1, int(round(_LOOKAHEAD_S * sample_rate)))
    rel = max(1, int(round(_RELEASE_S * sample_rate)))
    decay = float(np.exp(-1.0 / rel))
    block = max(block, 4 * la)
    stretches = _find_stretches(read, n, ceiling, 7 * rel + 4 * la, 2 * la, 7 * rel + 2 * la, block)
    pos = 0
    for s, e in stretches + [(n, n)]:
        for a in range(pos, s, block):
            yield a, np.asarray(read(a, min(s, a + block)), dtype=np.float32)
        yield from _limit_stretch(read, s, e, ceiling, la, decay, block)
        pos = e


def mix_blocks(
    stems: dict[str, np.ndarray],
    music_gain_db: float | None = None,
    ceiling: float = 0.99,
    sample_rate: int = 48000,
    block: int = _BLOCK,
) -> Iterator[tuple[int, np.ndarray]]:
    """`mix`, as (position, block) pairs in order, so no whole-length array is built.

    The stems may be memory-mapped files; only a few blocks are in memory at once.
    """
    _validate(stems)
    if music_gain_db is not None and music_gain_db > 0:
        raise ValueError("music_gain_db must be <= 0 (music is never made louder)")
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")
    g = None if music_gain_db is None else np.float32(10.0 ** (music_gain_db / 20.0))

    def read(a: int, b: int) -> np.ndarray:
        out = np.asarray(stems["speech"][:, a:b], dtype=np.float32) + np.asarray(
            stems["effects"][:, a:b], dtype=np.float32
        )
        if g is not None:
            out += np.asarray(stems["music"][:, a:b], dtype=np.float32) * g
        return out

    return limit_blocks(read, stems["speech"].shape[1], ceiling, sample_rate, block)


def mix(
    stems: dict[str, np.ndarray],
    music_gain_db: float | None = None,
    ceiling: float = 0.99,
    sample_rate: int = 48000,
    block: int = _BLOCK,
) -> np.ndarray:
    """Return speech + effects + music * gain, limited so it never exceeds ceiling.

    ``music_gain_db=None`` removes the music; a negative number keeps it that
    many dB quieter. Speech and effects are never rescaled.
    """
    blocks = mix_blocks(stems, music_gain_db, ceiling, sample_rate, block)  # validates when first used
    out = np.empty(stems["speech"].shape, np.float32)
    for a, y in blocks:
        out[:, a:a + y.shape[1]] = y
    return out


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


def _keep_spans(ranges: list[tuple[float, float]], n: int, sample_rate: int) -> list[list[int]]:
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
    return spans


def keep_blocks(
    blocks: Iterable[tuple[int, np.ndarray]],
    original: np.ndarray,
    ranges: list[tuple[float, float]],
    sample_rate: int = 48000,
    fade: float = 0.05,
) -> Iterator[tuple[int, np.ndarray]]:
    """`keep_ranges` over (position, block) pairs, e.g. from `mix_blocks`; yields the same
    kind of pairs. `original` may be a memory-mapped file."""
    if not hasattr(original, "shape"):
        original = np.asarray(original)
    if original.ndim != 2:
        raise ValueError("original must be a 2-D array")
    n = original.shape[1]
    spans = _keep_spans(ranges, n, sample_rate)
    fade_n = max(0, int(round(fade * sample_rate)))
    for start, mixed in blocks:
        stop = start + mixed.shape[1]
        out = mixed
        for a, b in spans:
            lo, hi = max(a, start), min(b, stop)
            if hi <= lo:
                continue
            if out is mixed:
                out = mixed.copy()
            f = min(fade_n, (b - a) // 2)
            out[:, lo - start:hi - start] = original[:, lo:hi]
            if f == 0:
                continue
            w = (np.arange(1, f + 1, dtype=np.float32) / np.float32(f + 1))[None, :]
            if a > 0 and (l2 := max(lo, a)) < (h2 := min(hi, a + f)):
                ww = w[:, l2 - a:h2 - a]
                out[:, l2 - start:h2 - start] = (mixed[:, l2 - start:h2 - start] * (1 - ww)
                                                 + original[:, l2:h2] * ww)
            if b < n and (l2 := max(lo, b - f)) < (h2 := min(hi, b)):
                ww = w[:, ::-1][:, l2 - (b - f):h2 - (b - f)]
                out[:, l2 - start:h2 - start] = (mixed[:, l2 - start:h2 - start] * (1 - ww)
                                                 + original[:, l2:h2] * ww)
        yield start, out


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
    out = np.empty_like(mixed)
    for a, y in keep_blocks([(0, mixed)], original, ranges, sample_rate, fade):
        out[:, a:a + y.shape[1]] = y
    return out
