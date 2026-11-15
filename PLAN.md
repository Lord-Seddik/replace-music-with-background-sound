# PLAN — music-remover

Status of this plan: **approved 2026-11-15**.

## Goal

A command-line tool that takes a video file (for example an anime episode),
removes the background music from its sound, keeps speech and sound effects,
and writes a new video file that is still pleasant to watch.

```
music-remover "Episode 01.mkv"      ->  "Episode 01.nomusic.mkv"
```

The input file is never modified. Video, subtitles and chapters are copied
as they are; only the sound is replaced.

## How it works

```
episode.mkv
   │  probe      find the audio tracks, pick one (ffprobe)
   │  extract    that track -> WAV (ffmpeg)
   │  separate   WAV -> speech / effects / music stems (neural model, GPU)
   │  mix        speech + effects (+ optional quiet music), match loudness
   │  mux        new sound + original video/subs/chapters -> output (ffmpeg)
   ▼
episode.nomusic.mkv
```

The separation model is the part that decides the quality. Ordinary "music
separation" models (vocals / drums / bass / other) are the wrong tool: they
put sound effects in the same stem as instruments. This project needs a
**cinematic** separation model, trained to split a film mix into
speech / music / effects. Which one is chosen by listening, in task T1,
before the rest is built around it.

### What "still enjoyable" means here

- Speech stays clear and sound effects (footsteps, hits, doors, ambience) stay.
- No audible "underwater" artefacts where music was removed. For scenes where
  the model struggles, a `--music-gain` option leaves the music very quiet
  instead of fully gone, which hides artefacts.
- Lips stay in sync: the output sound has exactly the input's length.
- Loudness of speech is the same as in the original.
- Openings and endings can be left untouched with `--keep 0:00-1:30`.

### Known limits (accepted for the first version)

- Sung vocals (insert songs, characters singing) may stay, since the model
  hears them as a voice.
- Effects that are musical (a chime, a sting) may be removed with the music.
- 5.1 sound is mixed down to stereo first.

## This machine

Checked on 2026-11-15:

| Thing   | Found                                   | Note |
|---------|-----------------------------------------|------|
| Python  | 3.12.10                                 | fine |
| FFmpeg  | 9.0.2 full build, with ffprobe          | fine |
| Git     | installed                               | fine |
| GPU     | NVIDIA RTX 3050 Laptop, 4 GB            | usable, but sound must be processed in short chunks |
| RAM     | 15 GB                                   | fine |
| Disk C: | about 15 GB free                        | **tight**: PyTorch with GPU support takes about 5 GB |

## Code layout

Following the style of the pasted `CONTRIBUTING.md` (small modules, each
with a narrow interface, tests that need no real media or model):

```
src/musicremover/
  probe.py       reads stream info; nothing else calls ffprobe
  extract.py     audio track -> WAV
  separate/      Separator interface + one backend per model
  mix.py         works on plain arrays, touches no files
  mux.py         builds the output file; nothing else writes video
  cli.py         the command
tests/           use short generated tones and a fake separator
samples/         real episodes for manual checks (ignored by Git)
output/          results (ignored by Git)
docs/
```

FFmpeg is always run with an argument list built by the code, never through
a shell string.

## Tasks

Tags and models follow `CLAUDE.md`: `[hard]` fable/opus, `[medium]` sonnet,
`[easy]` haiku.

### T1 — Choose the separation model `[hard]` — done
- Goal: pick the model by listening to real results.
- Work: cut three one-minute clips from a real episode (talk over music,
  action with effects over music, quiet scene). Run each candidate on them:
  a cinematic speech/music/effects model of the Bandit family trained on
  Divide-and-Remaster (the multilingual version if available, since the
  speech is Japanese), one alternative of the same kind, and Demucs as a
  baseline. Record speed and GPU memory use.
- Files: `experiments/` (throwaway scripts), `docs/model-choice.md`.
- Verify: you listen to the results in `output/bakeoff/` and agree with the
  choice written in `docs/model-choice.md`.
- Needs from you: one episode placed in `samples/`.
- Result so far: all three run on the 4 GB GPU. Recommended: Bandit v2
  multilingual (details in `docs/model-choice.md`). Chosen by you after
  listening, 2026-11-15.

### T2 — Project skeleton `[easy]` — done
- Goal: an installable empty package with tests running.
- Files: `pyproject.toml`, `src/musicremover/__init__.py`, `tests/`,
  `.gitignore`, `git init`.
- Verify: `pip install -e ".[dev]"` then `pytest` runs and passes.

### T3 — Probe and extract `[medium]` — done
- Goal: list a file's audio tracks and write one of them to WAV.
- Files: `probe.py`, `extract.py`, `tests/test_probe.py`, `tests/test_extract.py`.
- Verify: `pytest`; tests build a small video with FFmpeg's test sources and
  check track count, sample rate and length.

### T4 — Separation `[hard]` — done
- Goal: a `Separator` interface and the backend for the model chosen in T1,
  processing in overlapping chunks so it fits in 4 GB, with a CPU fallback.
  From T1: feed long audio in pieces with crossfaded overlaps (a whole episode
  at once crashes the GPU); default overlap 2, batch 1; copy only the Bandit v2
  model files from the runner (MIT) instead of importing the cloned repo; load
  weights with `weights_only=True`; weights are CC-BY-SA, so download, don't bundle.
- Files: `separate/__init__.py`, `separate/<model>.py`, `tests/test_separate.py`.
- Verify: `pytest` with a fake backend (chunk joins are seamless, output
  length equals input length to the sample); by hand, one clip from T1 gives
  ~~the same result as in the bake-off~~ **(changed during T4, please confirm)**
  a result that differs from the bake-off by no more than the bake-off differs
  from itself when its chunk grid is shifted by 1 s. An exact match is not
  possible: the model's output depends on where chunk edges fall.
- Result: 24 tests pass. Clip 1 vs bake-off: speech −27, music −17, effects
  −9 dB; the bake-off shifted by 1 s vs itself: speech −28, music −16,
  effects −8 dB. GPU 4.3× real time at overlap 2 (bake-off runner: 1.6×),
  1.2 GB peak. CPU fallback matches GPU (−79 dB) but runs at 0.14× real time.
  Files: `separate/__init__.py` (interface and chunking), `separate/bandit.py`,
  `separate/bandit_v2/` (model code copied from the runner, MIT, with notice).

### T5 — Mix `[medium]` — done
- Goal: speech + effects into the final sound, `--music-gain` option
  (default: music fully removed), speech loudness matched to the original,
  no clipping.
- Files: `mix.py`, `tests/test_mix.py`.
- Verify: `pytest` on generated arrays.

### T6 — Mux `[medium]` — done
- Goal: write the output file with the new sound as the default track,
  video/subtitles/chapters copied without re-encoding, optional
  `--keep-original` to keep the old sound as a second track.
- Files: `mux.py`, `tests/test_mux.py`.
- Verify: `pytest`; output has the expected streams, the same duration, and
  the input file is byte-for-byte unchanged.

### T7 — Command line `[medium]` — done
- Goal: `music-remover <file or folder>` running the whole pipeline with a
  progress display, temp-file cleanup, and refusal to overwrite existing files.
- Files: `cli.py`, `pyproject.toml` (entry point), `tests/test_cli.py`.
- Verify: `pytest` with the fake separator; by hand, a one-minute clip goes
  through end to end.

### T8 — Keep ranges `[medium]` — done
- Goal: `--keep START-END` (repeatable) leaves those parts of the sound
  original, with a short crossfade at the edges. For openings and endings.
- Files: `mix.py`, `cli.py`, tests.
- Verify: `pytest`; inside a kept range the output equals the input.

### T9 — Full-episode review `[hard]` — done
- Goal: run a whole episode, watch it, check sync, quality and run time, and
  review the code as a whole.
- Files: none expected; findings become new tasks here.
- Verify: you watch the result; findings are listed in this file.
- Done so far (2026-11-15): Clannad E20 run with the finished tool
  (`--keep-original`) → `output/t9/…nomusic.mkv`. 24.2 min in 5 min 45 s.
  Length identical to the sample; audio offset vs original 0 ms (correlation
  0.96); opening song −52 dB vs −22 dB original; overall level 1.4 dB lower
  (music gone, speech untouched); video, 2 subtitle tracks, 16 fonts and 7
  chapters kept. Code review findings: T11, T12.
- Your review: "sounds good mostly, good for a first version". Two problems:
  1. The dad's laugh around 20:30 is cut. Measured: 20:30.5–20:32.5 the model
     puts a loud sound (−22 dB) entirely in the *music* stem (speech −115 dB),
     so it is removed (output 31–39 dB below the original there). Short
     looks at the same window show similar but smaller dropouts at 20:16.5,
     20:21, 20:28.5 and 20:38.5. Listening files: `output/t9/laugh-2030/`.
     → T13.
  2. Some background sounds "not really perfect". No timestamps yet. → T14.

### T11 — Never delete a file mux did not create `[medium]` — todo
- Goal: `mux` deletes `out` when ffmpeg fails; if another program created a
  file at that path between the existence check and ffmpeg starting, it would
  be deleted. Write to a temporary name next to `out` and rename on success.
- Files: `mux.py`, `tests/test_mux.py`.
- Verify: `pytest`, including a test where ffmpeg fails and a pre-existing
  file at `out` survives.

### T12 — Lower peak memory for long files `[medium]` — todo
- Goal: separating a 24-minute episode briefly holds about 4 GB of RAM
  (stems, blending buffers, original). Fine on 15 GB, but a 2-hour film
  would need about 20 GB. Write finished stem parts out as soon as no later
  chunk can change them, instead of holding the whole episode.
- Files: `separate/__init__.py`, `cli.py`, tests.
- Verify: `pytest`; peak memory for the episode measured before and after.

### T13 — Rescue laughs and shouts the model calls music `[hard]` — todo
- Goal: non-speech voice sounds (laughs, shouts, gasps) sometimes land in the
  music stem and are removed. Find them and put them back without bringing
  back sung vocals in songs.
- Idea to test first (not decided): run a voice/instrument splitter (Demucs,
  already installed) on the *music stem only*. Where the music stem has a
  strong voice part but little instrumental part (a laugh alone; a song has
  both), return that voice part to the output. Compare against simpler
  options: a smaller default `--music-gain` around such moments, or none.
- Files: likely `separate/` (a second, optional pass), `mix.py`, tests; a
  short experiment script in `experiments/` first.
- Verify: on `output/t9/laugh-2030/` the laugh comes back (you listen); the
  opening song (clip 4) stays removed (level within 3 dB of today's −52 dB);
  speed cost measured.

### T14 — Background sounds that are "not really perfect" `[hard]` — todo
- Goal: find out what is wrong and whether it can be fixed.
- Needs from you: two or three timestamps where a background sound is off,
  and what you hear (cut, muffled, "underwater", too quiet…).
- Files: unknown until diagnosed; diagnosis first, then a plan entry.
- Verify: diagnosis written here per timestamp.

### T10 — Documentation `[easy]` — done
- Goal: a `README.md` with install and usage, and `CONTRIBUTING.md`
  rewritten for this project (the pasted one describes another project,
  fragdirector).
- Files: `README.md`, `CONTRIBUTING.md`.
- Verify: following the README on this machine works.

## Later ideas (not planned yet)

- Proper 5.1 handling (the centre channel carries most speech).
- A small window to drop files on, for use without a terminal.
- Automatic detection of openings and endings.

## Open questions

1. ~~Where should the environment live?~~ Answered: C:, in the project's
   `.venv`. Note: C: now has about 6 GB free; `output/` uses 6.4 GB.
2. Default output: same container as the input with `.nomusic` added to the
   name — is that right?
3. Is a command-line tool enough for the first version?
