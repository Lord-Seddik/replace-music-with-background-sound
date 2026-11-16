"""Fill the gaps the removed music leaves with ambience from a sound library.

The library is a folder with one sub-folder per scene type, named like the
scene labels (`classroom/`, `night_street/`, `park/`, ...), each holding
music-free recordings (wav, flac, ogg, mp3). Each gap is split at scene
cuts, each piece is labelled from video stills (`scenes.py`), and a matching
recording is laid under it, quietly, with crossfades where the scene changes.

Planning and rendering are pure numpy; `make_fill` ties them to the video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .gaps import Gap, find_gaps, frame_levels, split_at_cuts
from .scenes import (
    CLOSE_UP, DEFAULT_SCENES, SceneLabel, SceneModel, grab_frames, label_scenes, scene_cuts, still_times,
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
MIN_RECORDING_S = 1.0
_BLOCK = 1 << 20          # samples per step when rendering

BELOW_MUSIC_DB = 12.0       # fill sits this far below the music it replaces ...
LEVEL_RANGE_DB = (-55.0, -40.0)  # ... kept inside this range (dBFS, RMS)
EDGE_FADE_S = 1.0           # fade in/out where a gap starts or ends
SCENE_FADE_S = 0.5          # crossfade at a cut between two different scenes
LOOP_FADE_S = 2.0           # crossfade when a short recording is looped
JOIN_S = 0.6                # pieces this close in one gap count as touching
NEIGHBOUR_S = 30.0          # a close-up may borrow a scene from another gap this near

# Opening/ending songs (agreed 2026-11-16): long, almost no talking, near an end.
SONG_MIN_S = 60.0
SONG_MAX_TALK = 0.2
SONG_EDGE_S = 300.0
SONG_JOIN_S = 4.0           # a song also takes in neighbouring gaps this close ...
SONG_EDGE_TALK = 0.3        # ... with less talking than this (credits before/after it)
TALK_DB = -45.0             # speech stem above this = someone is talking (as in T15)

# When unsure of the scene, lay a neutral bed instead of a distinctive sound (T23,
# chosen against eye labels of Clannad E20: wrong fill fell from ~38 % to ~10 %).
NEUTRAL = "neutral"         # library folder with soft room tone/air; without it, unsure pieces stay empty
QUIET_SCENES = frozenset({"bedroom", "kitchen", "living_room", "shop", "classroom", "school_corridor",
                          "hospital", "train", "workplace"})  # room-like: used whenever chosen
DAY_SCENES = frozenset({"schoolyard", "quiet_street", "city_street", "park", "beach", "countryside", "forest"})
MIN_SCORE = 0.5             # any other scene needs this share of the model's match
DARK = 0.3                  # mean brightness (0-1) of a piece's stills below which it is night: no day sounds
# Choices offered to the model that never have a sound of their own: they only keep
# such shots from being mistaken for a scene that has one (snow looked like a beach).
LABEL_ONLY: dict[str, str] = {
    "snow": "a snowy landscape in winter",
    "title": "a title card with text on a plain background",
}
STEADY_TRIES = 8            # excerpt offsets tried per placement; the steadiest is used

# The neutral bed matches the kind of space (T25): `neutral_<space>` folders, falling
# back to `neutral`. The space is the one whose scenes together get the largest share
# of the model's match, so an unsure scene still gives a sure space; outdoor spaces
# become `night` on night shots. On Clannad E20 (eye labels) this cut indoor/outdoor
# mix-ups under the neutral bed from 330 s to 58 s.
SPACES = ("room", "hall", "town", "nature", "night")
OUTDOOR_SPACES = frozenset({"town", "nature", "night"})
SPACE_OF: dict[str, str] = {
    "bedroom": "room", "kitchen": "room", "living_room": "room", "shop": "room",
    "classroom": "hall", "school_corridor": "hall", "hospital": "hall", "workplace": "hall", "train": "hall",
    "city_street": "town", "quiet_street": "town", "schoolyard": "town", "night_street": "night",
    "park": "nature", "countryside": "nature", "forest": "nature", "beach": "nature", "snow": "nature",
}


@dataclass(frozen=True)
class Library:
    root: Path
    sounds: dict[str, tuple[Path, ...]]  # scene name -> recordings

    def scenes(self) -> dict[str, str]:
        """Scene descriptions for the model: every known scene type (with or without a
        folder, so a classroom is not mistaken for the nearest scene that has one), the
        label-only choices, the library's own folder names, and `close_up`."""
        out = dict(DEFAULT_SCENES) | LABEL_ONLY
        for n in self.sounds:
            if not _is_neutral(n) and n not in out:
                out[n] = "a " + n.replace("_", " ").replace("-", " ")
        return out


def _is_neutral(name: str) -> bool:
    return name == NEUTRAL or name.startswith(NEUTRAL + "_")


def space_of(label: SceneLabel, brightness: float) -> str | None:
    """The kind of space a piece shows (one of SPACES), None when no scene type
    with a known space gets any share (e.g. only custom folder names)."""
    sums: dict[str, float] = {}
    for name, share in label.shares.items():
        s = SPACE_OF.get(name)
        if s:
            sums[s] = sums.get(s, 0.0) + share
    if not sums:
        return None
    space = max(sums, key=sums.get)
    if space in OUTDOOR_SPACES and (label.night or brightness < DARK):
        space = "night"
    return space


def neutral_bed(space: str | None, library: Library) -> str:
    """The neutral folder for a space: `neutral_<space>`, else `neutral` (which may
    be missing from the library: then the piece stays empty)."""
    if space and f"{NEUTRAL}_{space}" in library.sounds:
        return f"{NEUTRAL}_{space}"
    return NEUTRAL


def choose_scene(label: SceneLabel, brightness: float, library: Library) -> str:
    """The folder to fill a piece from: its label when the library has it and the
    label is safe to use, `close_up` to borrow from a neighbour, else NEUTRAL."""
    name = label.label
    if name == CLOSE_UP:
        return CLOSE_UP
    if _is_neutral(name) or name not in library.sounds:
        return NEUTRAL
    if name not in QUIET_SCENES and label.score < MIN_SCORE:
        return NEUTRAL
    if brightness < DARK and name in DAY_SCENES:
        return NEUTRAL
    return name


def load_library(root: str | Path) -> Library:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"library folder not found: {root}")
    sounds = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name.lower()
        if name == CLOSE_UP:
            raise ValueError(f"{d}: '{CLOSE_UP}' is reserved (it means no place is visible)")
        files = tuple(sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS))
        if files:
            sounds[name] = files
    if not sounds:
        raise ValueError(f"no recordings in {root}: put them in one sub-folder per scene type, "
                         f"e.g. {root / 'classroom'}")
    return Library(root, sounds)


@dataclass(frozen=True)
class Placement:
    start: float
    end: float
    scene: str
    level_db: float  # target RMS of the fill, dBFS
    gap: int         # index of the gap it belongs to


@dataclass
class FillReport:
    placements: list[Placement] = field(default_factory=list)
    songs: list[Gap] = field(default_factory=list)     # gaps skipped as opening/ending songs
    unfilled: list[Gap] = field(default_factory=list)  # pieces with no visible place nearby


def talk_share(speech_db: np.ndarray, start: float, end: float, frame_s: float = 0.1) -> float:
    seg = speech_db[int(start / frame_s):max(int(start / frame_s) + 1, int(end / frame_s))]
    return float(np.mean(seg > TALK_DB)) if len(seg) else 0.0


def split_songs(gaps: list[Gap], speech_db: np.ndarray, duration: float) -> tuple[list[Gap], list[Gap]]:
    """(gaps to fill, gaps that look like the opening or ending song).

    A song is a long gap with almost no talking near either end. It then takes in
    neighbouring gaps closer than SONG_JOIN_S with little talking: the credits that
    start a little before the song gap or run on after it."""
    gaps = sorted(gaps, key=lambda g: g.start)
    talk = [talk_share(speech_db, g.start, g.end) for g in gaps]
    song = [(g.start < SONG_EDGE_S or g.end > duration - SONG_EDGE_S) and g.length >= SONG_MIN_S
            and t < SONG_MAX_TALK for g, t in zip(gaps, talk)]
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(gaps):
            if song[i] or talk[i] >= SONG_EDGE_TALK:
                continue
            before = i > 0 and song[i - 1] and g.start - gaps[i - 1].end <= SONG_JOIN_S
            after = i + 1 < len(gaps) and song[i + 1] and gaps[i + 1].start - g.end <= SONG_JOIN_S
            if before or after:
                song[i] = changed = True
    return [g for g, s in zip(gaps, song) if not s], [g for g, s in zip(gaps, song) if s]


def subtract_ranges(gaps: list[Gap], ranges: Sequence[tuple[float, float]], min_s: float = 2.0) -> list[Gap]:
    """Gaps with the given time ranges (e.g. --keep) cut out; leftovers under `min_s` dropped."""
    out = []
    for g in gaps:
        segs = [(g.start, g.end)]
        for a, b in ranges:
            nxt = []
            for s, e in segs:
                if b <= s or a >= e:
                    nxt.append((s, e))
                    continue
                if s < a:
                    nxt.append((s, a))
                if b < e:
                    nxt.append((b, e))
            segs = nxt
        out += [Gap(s, e, g.music_db, g.effects_db) for s, e in segs if e - s >= min_s]
    return out


def fill_level(music_db: float) -> float:
    lo, hi = LEVEL_RANGE_DB
    return round(min(hi, max(lo, music_db - BELOW_MUSIC_DB)), 1)


def _distance(p: Gap, q: Gap) -> float:
    return max(0.0, q.start - p.end, p.start - q.end)


def plan_fill(pieces: Sequence[tuple[int, Gap]], labels: Sequence[str]) -> tuple[list[Placement], list[Gap]]:
    """Placements in time order, and the pieces left unfilled.

    A `close_up` piece takes the scene of the nearest labelled piece in the
    same gap, else of one in another gap within NEIGHBOUR_S, else stays empty.
    Touching pieces of one gap with the same scene become one placement, so
    one recording runs on across the cut.
    """
    if len(pieces) != len(labels):
        raise ValueError("one label per piece")
    resolved: list[str | None] = []
    for (gi, p), own in zip(pieces, labels):
        if own != CLOSE_UP:
            resolved.append(own)
            continue
        best, best_key = None, None
        for (gj, q), lab in zip(pieces, labels):
            if lab == CLOSE_UP:
                continue
            d = _distance(p, q)
            if gj != gi and d > NEIGHBOUR_S:
                continue
            key = (gj != gi, d)
            if best_key is None or key < best_key:
                best, best_key = lab, key
        resolved.append(best)

    placements: list[Placement] = []
    unfilled: list[Gap] = []
    for (gi, p), scene in zip(pieces, resolved):
        if scene is None:
            unfilled.append(p)
            continue
        prev = placements[-1] if placements else None
        touching = prev is not None and prev.gap == gi and p.start - prev.end < JOIN_S
        if touching and prev.scene == scene:
            placements[-1] = Placement(prev.start, p.end, scene, prev.level_db, gi)
        else:
            start = prev.end if touching else p.start
            placements.append(Placement(start, p.end, scene, fill_level(p.music_db), gi))
    return placements, unfilled


def _ramp(n: int) -> np.ndarray:
    """Equal-power fade in over n samples (sine quarter), so crossfades of
    unrelated recordings keep a steady loudness."""
    return np.sin(0.5 * np.pi * (np.arange(n, dtype=np.float32) + 0.5) / max(n, 1)).astype(np.float32)


def _envelope(length: int, fade_in: int, fade_out: int, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Fade envelope of a `length`-sample piece, for samples [lo, hi) of it."""
    hi = length if hi is None else hi
    env = np.ones(hi - lo, np.float32)
    fade_in, fade_out = min(fade_in, length // 2), min(fade_out, length // 2)
    if fade_in and lo < fade_in:
        h = min(hi, fade_in)
        env[:h - lo] *= _ramp(fade_in)[lo:h]
    if fade_out and hi > length - fade_out:
        l = max(lo, length - fade_out)
        env[l - lo:] *= _ramp(fade_out)[::-1][l - (length - fade_out):hi - (length - fade_out)]
    return env


def _plan_excerpt(m: int, length: int, rng: np.random.Generator, loop_fade: int) -> tuple[int, list[tuple[int, int, bool]]]:
    """How `excerpt` builds `length` samples from a recording of `m` samples: the crossfade
    length and, for a short recording, its pieces as (position, start in the recording, first).
    A long enough recording is one piece at a random offset. The random numbers are
    drawn here, in the order `excerpt` always drew them."""
    if m >= length:
        return 0, [(0, int(rng.integers(0, m - length + 1)), True)]
    xf = max(1, min(loop_fade, m // 4))
    pieces, pos, first = [], 0, True
    while pos < length:
        s = 0 if first else int(rng.integers(0, m // 2 + 1))
        pieces.append((pos, s, first))
        pos += m - s - xf
        first = False
    return xf, pieces


def _excerpt_block(audio: np.ndarray, length: int, xf: int, pieces: list[tuple[int, int, bool]],
                   lo: int, hi: int) -> np.ndarray:
    """Samples [lo, hi) of the excerpt described by `_plan_excerpt`."""
    m = audio.shape[1]
    out = np.zeros((audio.shape[0], hi - lo), np.float32)
    if m >= length:
        s = pieces[0][1]
        out[:] = audio[:, s + lo:s + hi]
        return out
    for pos, s, first in pieces:
        size = m - s
        i0, i1 = max(lo, pos), min(hi, pos + min(size, length - pos))
        if i1 <= i0:
            continue
        chunk = audio[:, s + i0 - pos:s + i1 - pos].copy()
        k = np.arange(i0 - pos, i1 - pos)
        if not first:
            head = k < xf
            if head.any():
                chunk[:, head] *= _ramp(xf)[k[head]]
        tail = k >= size - xf
        if tail.any():
            chunk[:, tail] *= _ramp(xf)[::-1][k[tail] - (size - xf)]
        out[:, i0 - lo:i1 - lo] += chunk
    return out


def excerpt(audio: np.ndarray, length: int, rng: np.random.Generator, loop_fade: int) -> np.ndarray:
    """`length` samples from a random place in `audio`; a short recording is
    looped from random starts with crossfades."""
    xf, pieces = _plan_excerpt(audio.shape[1], length, rng, loop_fade)
    return _excerpt_block(audio, length, xf, pieces, 0, length)


def _to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return np.repeat(audio, channels, axis=0)
    if channels == 1:
        return audio.mean(axis=0, keepdims=True)
    return audio[:channels] if audio.shape[0] > channels else np.concatenate(
        [audio, np.repeat(audio[-1:], channels - audio.shape[0], axis=0)])


def read_recording(path: Path, sample_rate: int, channels: int) -> tuple[np.ndarray, float]:
    """(audio as float32 (channels, samples) at `sample_rate`, its level in dB RMS).
    The level ignores near-silent stretches so a quiet tail does not lower it."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data.T)
    if sr != sample_rate:
        import torch
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, sample_rate).numpy()
    audio = _to_channels(audio, channels).astype(np.float32, copy=False)
    if audio.shape[1] < MIN_RECORDING_S * sample_rate:
        raise ValueError(f"{path}: shorter than {MIN_RECORDING_S:g} s")
    levels = frame_levels(audio, sample_rate)
    active = levels[levels > -70]
    if not len(active):
        raise ValueError(f"{path}: silent")
    return audio, float(10 * np.log10(np.mean(10 ** (active / 10))))


def render_fill(
    placements: Sequence[Placement],
    library: Library,
    channels: int,
    length: int,
    sample_rate: int = 48000,
    seed: int = 0,
    read: Callable[[Path, int, int], tuple[np.ndarray, float]] = read_recording,
    level_offset_db: float = 0.0,
    out: np.ndarray | None = None,
    block: int = _BLOCK,
    steady: bool = False,
) -> np.ndarray:
    """The fill track (channels, length): each placement gets a recording of its
    scene at its level (plus `level_offset_db`), faded in/out at gap edges and
    crossfaded where two placements touch (each runs SCENE_FADE_S/2 past the cut).

    Placements are rendered `block` samples at a time. `out` may be a zero-filled
    (channels, length) float32 array such as a memory-mapped file; the fill is
    added into it, and no array longer than a block is made.

    With `steady`, a recording longer than the placement is cut where it is
    steadiest (of STEADY_TRIES random offsets, the one whose loudest moment stands
    least above its median), so a shout or a horn in the recording is avoided.
    """
    if out is None:
        out = np.zeros((channels, length), np.float32)
    rng = np.random.default_rng(seed)
    cache: dict[Path, tuple[np.ndarray, float]] = {}
    level_cache: dict[Path, np.ndarray] = {}
    edge, scene_fade = int(EDGE_FADE_S * sample_rate), int(SCENE_FADE_S * sample_rate)
    half = SCENE_FADE_S / 2
    for k, pl in enumerate(placements):
        before = k > 0 and abs(placements[k - 1].end - pl.start) < 1e-6
        after = k + 1 < len(placements) and abs(placements[k + 1].start - pl.end) < 1e-6
        a = max(0, int(round((pl.start - (half if before else 0)) * sample_rate)))
        b = min(length, int(round((pl.end + (half if after else 0)) * sample_rate)))
        if b <= a:
            continue
        files = library.sounds[pl.scene]
        path = files[int(rng.integers(len(files)))]
        if path not in cache:
            cache[path] = read(path, sample_rate, channels)
        audio, level = cache[path]
        if steady and audio.shape[1] >= b - a:
            if path not in level_cache:
                level_cache[path] = frame_levels(audio, sample_rate)
            xf, pieces = 0, [(0, _steadiest(level_cache[path], audio.shape[1], b - a, sample_rate, rng), True)]
        else:
            xf, pieces = _plan_excerpt(audio.shape[1], b - a, rng, int(LOOP_FADE_S * sample_rate))
        gain = np.float32(10 ** ((pl.level_db + level_offset_db - level) / 20))
        fade_in, fade_out = scene_fade if before else edge, scene_fade if after else edge
        for lo in range(0, b - a, block):
            hi = min(b - a, lo + block)
            seg = _excerpt_block(audio, b - a, xf, pieces, lo, hi)
            out[:, a + lo:a + hi] += seg * gain * _envelope(b - a, fade_in, fade_out, lo, hi)
    return out


def _steadiest(levels: np.ndarray, m: int, length: int, sample_rate: int, rng: np.random.Generator,
               frame_s: float = 0.1) -> int:
    """Start (in samples) of the steadiest `length`-sample window of a recording of
    `m` samples, among STEADY_TRIES random starts; `levels` are its frame levels (dB)."""
    starts = rng.integers(0, m - length + 1, size=STEADY_TRIES)
    best, best_key = int(starts[0]), None
    for s in starts:
        a = int(s / sample_rate / frame_s)
        seg = levels[a:max(a + 1, int((s + length) / sample_rate / frame_s))]
        key = float(seg.max() - np.median(seg))
        if best_key is None or key < best_key:
            best, best_key = int(s), key
    return best


def add_fill(effects: np.ndarray, fill: np.ndarray, block: int = _BLOCK) -> None:
    """effects += fill, a block at a time (both may be memory-mapped files)."""
    for a in range(0, effects.shape[1], block):
        effects[:, a:a + block] += fill[:, a:a + block]


def make_fill(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    video: str | Path,
    library: Library,
    model: SceneModel | Callable[[], SceneModel],
    keep: Sequence[tuple[float, float]] = (),
    progress: Callable[[str], None] = lambda stage: None,
    cuts: Sequence[float] | None = None,
    level_offset_db: float = 0.0,
    allocate: Callable[[tuple[int, int]], np.ndarray] | None = None,
) -> tuple[np.ndarray | None, FillReport]:
    """The fill track for an episode (None when there is nothing to fill).
    `model` may be a factory, so the image model is only loaded when needed.
    `allocate(shape)` makes the zero-filled array the fill is rendered into (default:
    in memory; the CLI passes a memory-mapped file)."""
    report = FillReport()
    channels, length = stems["music"].shape
    duration = length / sample_rate
    progress("finding gaps")
    gaps = find_gaps(stems, sample_rate)
    speech_db = frame_levels(stems["speech"], sample_rate)
    gaps, report.songs = split_songs(gaps, speech_db, duration)
    gaps = subtract_ranges(gaps, keep)
    if not gaps:
        return None, report
    if cuts is None:
        progress("finding scene cuts")
        cuts = scene_cuts(video)
    pieces = [(gi, p) for gi, g in enumerate(gaps) for p in split_at_cuts([g], list(cuts))]
    if not pieces:
        return None, report
    progress("grabbing stills")
    times = [still_times(p.start, p.end) for _, p in pieces]
    flat = grab_frames(video, [t for ts in times for t in ts])
    stills, i = [], 0
    for ts in times:
        stills.append(flat[i:i + len(ts)])
        i += len(ts)
    progress("labelling scenes")
    if not hasattr(model, "scores"):
        model = model()
    labels = label_scenes(stills, model, library.scenes(), day_night=True)
    brightness = [float(np.mean([s.mean() for s in group])) / 255 for group in stills]
    chosen = []
    for lb, b in zip(labels, brightness):
        scene = choose_scene(lb, b, library)
        chosen.append(neutral_bed(space_of(lb, b), library) if scene == NEUTRAL else scene)
    report.placements, report.unfilled = plan_fill(pieces, chosen)
    # a neutral bed the library does not have: leave those pieces empty
    report.unfilled += [Gap(p.start, p.end, 0.0, 0.0) for p in report.placements if p.scene not in library.sounds]
    report.placements = [p for p in report.placements if p.scene in library.sounds]
    if not report.placements:
        return None, report
    progress("laying ambience")
    return render_fill(report.placements, library, channels, length, sample_rate,
                       level_offset_db=level_offset_db, steady=True,
                       out=None if allocate is None else allocate((channels, length))), report
