"""Command line: remove background music from video files."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .extract import extract
from .mix import keep_ranges, mix, parse_range
from .mux import mux
from .probe import ProbeError, choose_track, probe
from .separate import separate
from .separate.bandit import WEIGHTS_URL, BanditV2

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".mov", ".webm"}
SAMPLE_RATE = 48000
DEFAULT_CACHE_WEIGHTS = Path.home() / ".cache" / "music-remover" / "checkpoint-multi.ckpt"


def _range_arg(text: str) -> tuple[float, float]:
    try:
        return parse_range(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="music-remover",
        description="Remove background music from videos, keeping speech and sound effects.",
    )
    p.add_argument("inputs", nargs="+", help="video files or folders of videos")
    p.add_argument("-o", "--out-dir", type=Path,
                   help="where to write outputs (default: next to each input)")
    p.add_argument("--language",
                   help="audio track language to use, e.g. jpn (default: the default track)")
    p.add_argument("--music-gain", type=float, metavar="DB",
                   help="keep the music this many dB quieter (negative, e.g. -20) instead of removing it")
    p.add_argument("--keep", action="append", type=_range_arg, metavar="START-END",
                   help="leave this part of the soundtrack original (repeatable), "
                        "e.g. --keep 0:00-1:30 --keep 22:10-end")
    p.add_argument("--keep-original", action="store_true", help="also keep the original audio tracks")
    p.add_argument("--quality", choices=("normal", "high"), default="normal",
                   help="high is about twice as slow")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--weights", type=Path, help="model weights file")
    return p


def _resolve_weights(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    env = os.environ.get("MUSICREMOVER_WEIGHTS")
    if env:
        return Path(env)
    local = Path("models") / "bandit_v2_multi.ckpt"
    if local.is_file():
        return local
    return DEFAULT_CACHE_WEIGHTS


def _collect(inputs: list[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += sorted(
                (c for c in p.iterdir()
                 if c.is_file() and c.suffix.lower() in VIDEO_EXTENSIONS
                 and ".nomusic." not in c.name),
                key=lambda c: c.name,
            )
        elif p.is_file():
            files.append(p)
        else:
            errors.append(f"not found: {p}")
    return files, errors


def _out_path(src: Path, out_dir: Path | None) -> Path:
    return (out_dir or src.parent) / f"{src.stem}.nomusic{src.suffix}"


def _fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class _Line:
    """One updating status line on stderr."""

    def __init__(self) -> None:
        self.width = 0

    def show(self, text: str) -> None:
        pad = " " * max(0, self.width - len(text))
        sys.stderr.write("\r" + text + pad)
        sys.stderr.flush()
        self.width = len(text)

    def done(self) -> None:
        if self.width:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.width = 0


def _process(src: Path, out: Path, args: argparse.Namespace, separator, tag: str) -> None:
    line = _Line()
    prefix = f"{tag} {src.name}"
    start = time.monotonic()
    try:
        line.show(f"{prefix}  probing")
        info = probe(src)
        track = choose_track(info, args.language)
        with tempfile.TemporaryDirectory(prefix="musicremover-") as tmp:
            tmp_dir = Path(tmp)
            line.show(f"{prefix}  extracting")
            wav = extract(src, track, tmp_dir / "in.wav", SAMPLE_RATE)
            data, _sr = sf.read(wav, dtype="float32", always_2d=True)
            audio = np.ascontiguousarray(data.T)
            del data

            t0 = time.monotonic()

            def progress(done: int, total: int) -> None:
                elapsed = time.monotonic() - t0
                eta = elapsed / done * (total - done) if done else 0.0
                line.show(f"{prefix}  separating {100 * done // total}%  eta {_fmt_time(eta)}")

            stems = separate(separator, audio,
                             overlap=2 if args.quality == "normal" else 4,
                             progress=progress)
            line.show(f"{prefix}  mixing")
            mixed = mix(stems, music_gain_db=args.music_gain, sample_rate=SAMPLE_RATE)
            del stems
            if args.keep:
                mixed = keep_ranges(mixed, audio, args.keep, SAMPLE_RATE)
            del audio
            line.show(f"{prefix}  writing")
            new_wav = tmp_dir / "out.wav"
            sf.write(new_wav, np.ascontiguousarray(mixed.T), SAMPLE_RATE, subtype="FLOAT")
            del mixed
            out.parent.mkdir(parents=True, exist_ok=True)
            mux(src, new_wav, out, keep_original=args.keep_original, language=track.language)
        line.show(f"{prefix}  done")
        line.done()
        print(f"{out}  ({_fmt_time(time.monotonic() - start)})")
    except BaseException:
        line.done()
        raise


def main(argv: list[str] | None = None, separator_factory=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.music_gain is not None and args.music_gain > 0:
        print("error: --music-gain must be 0 or negative", file=sys.stderr)
        return 2

    files, errors = _collect(args.inputs)
    failed = 0
    for e in errors:
        print(f"failed: {e}", file=sys.stderr)
        failed += 1
    if not files:
        if failed:
            return 1
        print("error: no video files found in the given inputs", file=sys.stderr)
        return 2

    jobs = []
    for src in files:
        out = _out_path(src, args.out_dir)
        if out.exists():
            print(f"skipped (exists): {out}")
        else:
            jobs.append((src, out))

    weights = _resolve_weights(args.weights)
    if jobs and separator_factory is None and not weights.is_file():
        print(f"error: model weights not found at {weights}\n"
              f"Download checkpoint-multi.ckpt from {WEIGHTS_URL}\n"
              f"and save it as {Path('models') / 'bandit_v2_multi.ckpt'} "
              f"or {DEFAULT_CACHE_WEIGHTS},\n"
              f"or pass --weights PATH / set MUSICREMOVER_WEIGHTS.", file=sys.stderr)
        return 2

    separator = None
    try:
        for i, (src, out) in enumerate(jobs, 1):
            try:
                if separator is None:
                    separator = (separator_factory or BanditV2)(weights, args.device)
                _process(src, out, args, separator, f"[{i}/{len(jobs)}]")
            except (ProbeError, RuntimeError, OSError, ValueError) as exc:
                print(f"failed: {src}: {exc}", file=sys.stderr)
                failed += 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 1 if failed else 0
