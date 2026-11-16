"""Find the gaps the removed music leaves (pure numpy, no files).

A gap is a stretch where the music was audible and was the main non-speech
sound, so that after removing it the background is far quieter than before.
Thresholds were measured on the test episode in T15 (see PLAN.md): music
above -50 dB and at least 15 dB above the effects stem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_S = 0.1
SMOOTH_S = 1.0
AUDIBLE_DB = -50.0
MARGIN_DB = 15.0
MIN_GAP_S = 2.0
BRIDGE_S = 1.0


@dataclass(frozen=True)
class Gap:
    start: float  # seconds
    end: float
    music_db: float    # level of the removed music in this gap
    effects_db: float  # level of the kept background in this gap

    @property
    def length(self) -> float:
        return self.end - self.start


def frame_levels(audio: np.ndarray, sample_rate: int, frame_s: float = FRAME_S) -> np.ndarray:
    """Power per frame in dB, averaged over channels; the last partial frame is padded."""
    frame = max(1, int(frame_s * sample_rate))
    length = audio.shape[1]
    count = max(1, -(-length // frame))
    power = np.zeros(count)
    for i in range(0, count, 1 << 8):  # blocks keep float64 copies small (about 1.2M samples)
        a, b = i * frame, min(length, (i + (1 << 8)) * frame)
        block = audio[:, a:b].astype(np.float64)
        n = -(-(b - a) // frame)
        padded = np.zeros((audio.shape[0], n * frame))
        padded[:, :b - a] = block
        power[i:i + n] = (padded ** 2).reshape(audio.shape[0], n, frame).mean(axis=(0, 2))
    return 10 * np.log10(power + 1e-12)


def _smooth(levels_db: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return levels_db
    power = 10 ** (levels_db / 10)
    return 10 * np.log10(np.convolve(power, np.ones(frames) / frames, mode="same") + 1e-12)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def find_gaps(
    stems: dict[str, np.ndarray],
    sample_rate: int = 48000,
    *,
    audible_db: float = AUDIBLE_DB,
    margin_db: float = MARGIN_DB,
    min_gap_s: float = MIN_GAP_S,
    bridge_s: float = BRIDGE_S,
    frame_s: float = FRAME_S,
    smooth_s: float = SMOOTH_S,
) -> list[Gap]:
    """Gaps in time order. Holes shorter than `bridge_s` inside a gap are joined;
    gaps shorter than `min_gap_s` are dropped."""
    music = _smooth(frame_levels(stems["music"], sample_rate, frame_s), round(smooth_s / frame_s))
    effects = _smooth(frame_levels(stems["effects"], sample_rate, frame_s), round(smooth_s / frame_s))
    mask = (music > audible_db) & (music - effects >= margin_db)

    bridge = round(bridge_s / frame_s)
    runs = _runs(mask)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < bridge:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    duration = stems["music"].shape[1] / sample_rate
    gaps = []
    for a, b in merged:
        if (b - a) * frame_s < min_gap_s:
            continue
        m = 10 * np.log10(np.mean(10 ** (music[a:b] / 10)))
        e = 10 * np.log10(np.mean(10 ** (effects[a:b] / 10)))
        gaps.append(Gap(round(a * frame_s, 3), round(min(b * frame_s, duration), 3), round(m, 1), round(e, 1)))
    return gaps


def split_at_cuts(gaps: list[Gap], cuts: list[float], min_gap_s: float = 0.5) -> list[Gap]:
    """Split gaps at scene cuts so each piece lies inside one scene.
    Pieces shorter than `min_gap_s` (a cut right at a gap edge) are dropped."""
    cuts = sorted(cuts)
    out = []
    for g in gaps:
        edges = [g.start] + [c for c in cuts if g.start < c < g.end] + [g.end]
        for a, b in zip(edges, edges[1:]):
            if b - a >= min_gap_s:
                out.append(Gap(a, b, g.music_db, g.effects_db))
    return out
