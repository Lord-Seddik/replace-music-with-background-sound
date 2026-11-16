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

### T11 — Never delete a file mux did not create `[medium]` — done
- Goal: `mux` deletes `out` when ffmpeg fails; if another program created a
  file at that path between the existence check and ffmpeg starting, it would
  be deleted. Write to a temporary name next to `out` and rename on success.
- Files: `mux.py`, `tests/test_mux.py`.
- Verify: `pytest`, including a test where ffmpeg fails and a pre-existing
  file at `out` survives.

### T12 — Lower peak memory for long files `[medium]` — done
- Goal: separating a 24-minute episode briefly holds about 4 GB of RAM
  (stems, blending buffers, original). Fine on 15 GB, but a 2-hour film
  would need about 20 GB. Write finished stem parts out as soon as no later
  chunk can change them, instead of holding the whole episode.
- Files: `separate/__init__.py`, `cli.py`, tests.
- Verify: `pytest`; peak memory for the episode measured before and after.
- Done (2026-11-16): `separate()` keeps blending buffers for one chunk only
  and writes each finished stretch to `out=` (new `disk_stems()` makes
  memory-mapped files; default is still in-memory arrays); output is
  bit-identical to the old `separate()` on 56 test cases. `cli.py` writes the
  stems to the temp folder, reads/writes the WAV in blocks, and clears the
  traceback frames before deleting the folder (Windows cannot delete mapped
  files). 174 tests pass (3 new).
- Measured on Clannad E20, default options, same run before and after
  (`output/t12/`): peak working set 5.09 → 4.45 GiB, peak commit 10.62 →
  9.06 GiB (about 2.4 GiB of that is Python + torch + CUDA before any audio is
  loaded). Separating alone went from +4.97 to +2.64 GiB, but voice rescue
  and the mix still build whole-episode arrays, so they now set the peak.
  Final audio of the two runs differs by −70 dB (GPU run-to-run noise); run
  time unchanged (7:23 vs 7:21).
- **Not fully met:** a 2-hour film would still need roughly 4× today's peak.
  Fixing that needs `voice.py` and `mix.py` (T20).

### T20 — Bring voice rescue and mix down to a few chunks of memory `[medium]` — done
- Goal: `rescue_voices`, `mix` (limiter, `keep_ranges`) and the library fill
  work in blocks or on the disk-backed stems, so peak memory no longer grows
  with episode length. Found while measuring T12.
- Files: `separate/voice.py`, `mix.py`, `library.py`, `cli.py`, tests.
- Verify: `pytest` (results identical to the whole-array versions on
  generated signals); peak working set for the episode measured again; a
  synthetic 2-hour signal stays within a limit chosen when the task starts.
- Done (2026-11-16, Sonnet agent; reviewed, 220 tests pass here): limiter,
  `--keep`, voice rescue and the fill all work in blocks (`mix_blocks`,
  `limit_blocks`, `keep_blocks`; `rescue_voices(out=, scratch=)` in place
  on the disk stems; `render_fill(out=)`, `add_fill`); `cli.py` mixes,
  applies `--keep` and writes the WAV in one streaming pass. Also
  `gaps.frame_levels` blocks cut from ~78M to ~1.2M samples (it made 3.8 GB
  of temporaries). `tests/test_blocks.py` (46 tests) compares against frozen
  copies of the old code: bit-identical (keep, fill) or within 2e-6.
- Limit chosen before measuring: 2-hour synthetic run's peak commit at most
  1 GiB above a 1-minute run. Measured 0.42 vs 0.40 GiB: met
  (`experiments/long_synthetic.py`).
- Episode (`output/t20/*.log`), before → after: default options, peak
  commit 9.06 → 5.27 GiB, working set 6.40 → 4.54 GiB, 7:03 → 7:35; with
  the fill, commit 9.59 → 5.29 GiB, working set 6.92 → 5.26 GiB, 11:32 →
  10:24. Final audio before vs after −39 dB, but old code vs itself also
  differs by −39 dB (GPU noise in voice rescue near its thresholds, ~70 s
  around 4:20 and 20:30–22:30); the other seconds are bit-identical.
- **Not fully met:** working set on the 2-hour synthetic still reaches
  7.75 GiB because Windows keeps memory-mapped pages resident until it
  trims them (they can be reclaimed; commit does not grow). → T22.
- Found, not fixed: a 2-hour film needs ~20 GB of temporary disk (float
  WAV, `audio.npy`, stems, out WAV) → T22; voice rescue is not repeatable
  run to run near its thresholds (minor).

### T22 — Long films: temporary disk and working set `[medium]` — done
- Goal: a 2-hour film needs ~20 GB of temp disk and its working set grows
  with length through memory-mapped files (from T20). Use plain block
  reads/writes instead of memmaps where they are only streamed, and drop
  copies that are not needed (e.g. both the float WAV and `audio.npy`).
- Files: `cli.py`, `separate/__init__.py`, `extract.py`, tests.
- Verify: `pytest`; `experiments/long_synthetic.py 120` working set within
  1 GiB of the 1-minute run; temp disk for the episode measured before and
  after; final audio unchanged on deterministic input.
- Done (2026-11-16, Sonnet agent; reviewed, 227 tests pass here): stems,
  fill and voice scratch are `FileArray`s (plain raw files, read/written
  in blocks, no memmaps) in `separate/__init__.py`; `extract.WavArray`
  reads slices from `in.wav`, so the `audio.npy` copy is gone; `in.wav` is
  deleted right after separation unless `--keep` needs it; `mix.py` no
  longer forces whole-file reads via `np.asarray`. `tests/test_fileio.py`
  (7 tests, bit-identical to the in-memory paths).
- Measured: 2-hour synthetic working set 0.08 GiB vs 0.06 GiB for 1 minute
  (was 7.75), commit 0.41 vs 0.40 GiB. Episode, before → after: temp disk
  4.15 → 2.60 GiB, working set 5.26 → 2.27 GiB, commit 5.21 → 5.15 GiB,
  run time 428 → 441 s. Final audio vs the T20 run: 1360 of 1451 seconds
  bit-identical; the rest is the known voice-rescue GPU noise (old code
  vs T20 differs in the same ~70 s). Logs in `output/t22/`.
- Left (minor, not planned): mux still needs a finished `out.wav`
  (~2.8 GiB for a 2-hour film); piping to ffmpeg would remove it.

### T13 — Rescue laughs and shouts the model calls music `[hard]` — done
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
- Built: `separate/voice.py` (Demucs htdemucs on the music stem only; frames
  of 0.1 s; voice moved back where it is 20–30 dB or more above the
  instruments, smoothly), `--no-voice-rescue`, `separate()` now takes
  fractional overlap and stem names; `demucs` added to dependencies. 129
  tests pass. Experiment: `experiments/voice_rescue.py`.
- Measured on the full episode (`output/t13/episode/`): laugh at 20:30
  −20.7 dB (original −20.6, before −61.7), its tail −23.9 (original −23.8,
  before −55.9); opening song −52.4 dB (unchanged); sound added back in only
  26 of 1451 seconds. Run time 7:06 instead of 5:45 (+1:21).
- Side effect: ~10 short moments of the ending song came back quietly
  (−30 to −48 dB, original −27) at 22:10–22:17, 22:27, 23:34 — probably
  moments where the singer is briefly alone. Before/after files:
  `output/t13/compare/`.
- Waiting for: you to listen (laugh, and whether the ending leak bothers you).
- Your listening (2026-11-16): the laugh comes back ("i guess there is a
  laugh yes"). Ending leak: not raised; kept as a known limit (reopen if
  it bothers you).

### Finishing order (your request 2026-11-16: "do the next tasks you can,
### give the others to other models, finish everything")
1. T20 `[medium]` → Sonnet agent (code, tests, memory measurement).
2. CC0 starter library downloaded by Claude (you asked for it instead of
   downloading by hand) into `samples/ambience/`.
3. T18b by-hand check `[hard]` (Opus): episode with the real library, after
   T20 so the run also checks T20; clips for you.
4. T17 `[hard]` → Opus agent, experiments only, keeping ≥ 4 GB free on C:.
GPU runs happen one at a time (4 GB card).

### T14 — Background sounds that are "not really perfect" `[hard]` — done
- Goal: find out what is wrong and whether it can be fixed.
- ~~Needs from you: timestamps~~ Searched the episode instead (your choice,
  2026-11-16). Scripts: `experiments/background_scan.py` (band levels per
  0.1 s for original, stems, T13 output) and `experiments/background_find.py`
  (lists suspect moments; run with `output/t14/features.npz`). Data and spectrograms (rows: original, music,
  effects, speech, output) in `output/t14/`.
- Files: unknown until diagnosed; diagnosis first, then a plan entry.
- Verify: diagnosis written here per timestamp.
- Ruled out: the mix stage (output = speech + effects within 0.5 dB in 95 %
  of frames; limiter barely acts); lost sound (stems add up to the original
  within −41 dB median); a wrong level where there is no music (output −
  original = 0.0 dB median). Noise-like sounds (wind, rain) in the music
  stem appear only inside the two songs.
- Found — the problem is how the model splits sound **while music plays**:
  1. **Sound tails cut off.** With music present, kept sounds die away about
     twice as fast as without (median 22 vs 11 dB in 0.5 s; slowest quarter
     40 vs 15 dB). The model gives the echo/decay of a sound to the music
     stem, so steps and hits stop dead in the output.
  2. **Sounds pop out of silence.** With music and no talking (378 s), the
     effects stem is near-silent (< −85 dB) 60 % of the time; without music
     only 19 %. Kept sounds start and stop against digital silence instead of
     a room. (Same cause as the Stage 2 gaps; the library fill partly hides it.)
  3. **Crowd/room background switches on and off in blocks.** A steady
     background (school corridor talk, ballfield crowd) is split between the
     effects and music stems and the share flips every second or two, so it
     comes and goes in the output.
- Per timestamp (clips play original, 1 s black, then output):
  - 3:37–4:07, robot scene: clanks with cut tails in silence (1, 2).
    `output/t14/1_robot_03-37_original-then-output.mkv`
  - 9:00–9:14, school corridor: background talk and room sound come and go
    in blocks; shifts into the speech stem when someone talks (3).
    `output/t14/2_corridor_09-00_original-then-output.mkv`
  - 21:42–22:02, ballfield: crowd sound starts and stops abruptly; longest
    music-plus-background stretch outside the songs (3, 1).
    `output/t14/3_ballfield_21-42_original-then-output.mkv`
  - Not a fault: 0:03.5–0:06.6, a shout the model called music is put back
    fully by the T13 voice rescue.
- Possible fixes (not planned until you confirm which problem you heard):
  a. Tail keeper: right after a kept sound, keep the music stem's energy
     briefly under the same decay as the sound (short release, ~0.3–0.5 s),
     only where the music stem is not tonal. Small, local, testable.
  b. Smooth the background: give the effects stem a slow floor follower
     (hold its level a few seconds instead of letting it drop to silence)
     in scenes where it was flipping. Risk: brings back a little music.
  c. Use the existing `--music-gain` (e.g. −24 dB) to hide all three at the
     cost of faint music; no code needed, cheap to try first.
- Fixes tried (2026-11-16, `experiments/tail_keeper.py`, on the T1 scan
  stems, nothing in `src/` changed). a and b are one mechanism: per
  frequency bin, follow the effects stem's level (instant rise, natural
  fall of 24 dB/s), and put back music-stem sound only under that level,
  at most 15 dB below it, never from sharp peaks (notes). "tails" falls at
  once; "hold" keeps the level 2 s before falling.
- Measured on the three spots together (robot, corridor, ballfield), and on
  60 s of the opening song (2:20–3:20, must stay within 3 dB of today):

  | Option | Fall in 0.5 s (median) | Near-silent time | Opening song |
  |--------|------------------------|------------------|--------------|
  | today  | 23 dB                  | 21 %             | −52.0 dB     |
  | tails  | 18 dB                  | 8 %              | −51.4 dB     |
  | hold   | 11 dB (= no-music scenes) | 0.2 %         | −50.5 dB     |
  | `--music-gain -24` | 14 dB      | 0.3 %            | −46.7 dB     |

  Original song level is about −22 dB. "hold" matches the natural fall and
  removes the silence while keeping the song removed; `--music-gain -24`
  does the same but lets the whole music back at −24 dB everywhere.
  Caps of −10/−20 dB were also measured (−10 brings back more, −20 less).
- Listening clips, one file per option, in `output/t14/fixes/`:
  `<spot>_0_original`, `_1_now`, `_2_gain24`, `_3_tails`, `_4_hold`;
  songs (`4_opening-song`, `5_ending-song`, 22:40 −51.7 vs −52.4 today)
  only original/now/hold. These use the T1 stems, without the T13 rescue.
- Your decision (2026-11-16): build "hold". A faint trace of music is
  acceptable as long as the music is **not recognisable**. → T21. You had
  not listened to the fix clips yet; that listening check is part of T21.
- Your listening (2026-11-16): `_1_now` "seems good, few stuff, but seems
  good". `_4_hold` "has like direct music": filling the silence feels
  good, but the music is recognisable, so it fails your rule.

### T21 — Keep the background that the model files under music `[hard]` — dropped
- **Dropped 2026-11-16 (your decision):** you heard
  recognisable music in the "hold" clips and found today's output good.
  Taking sound back out of the music stem is what lets the music in, and
  tightening it ("tails" or a −20 dB cap) brings back less of both. The
  empty feeling is already what the library fill (T18b) is for, using sound
  that has no music in it at all.
- Goal: the T14 "hold" pass as a pipeline step, on by default: per
  frequency bin, follow the effects stem's level (instant rise, hold 2 s,
  then fall 24 dB/s) and move music-stem sound back to the effects stem only
  under that level, at most 15 dB below it, never from sharp spectral peaks
  (notes). Option `--no-background-keep` turns it off.
- Rule from you: music may leave a faint trace but must not be recognisable.
- Design points:
  - Runs after voice rescue, on what is left of the music stem (so a
    rescued voice is not added twice), before the library fill and mix.
  - Works in blocks with its state (level, hold age) carried across block
    edges, so it fits T20 (memory must not grow with episode length).
  - Settings (hold 2 s, fall 24 dB/s, cap −15 dB, peak test 4× the median
    of 9 neighbouring bins) start from the T14 experiment; change only if
    the episode check fails.
- Files: new `src/musicremover/separate/background.py`, `cli.py` (step and
  flag), `tests/test_background.py`, README (one line on the option).
- Verify:
  - `pytest`: a noise burst whose tail sits in the music stem gets its tail
    back; a steady tone in the music stem is never put back; block-wise
    result equals the whole-array result; output length unchanged; off
    switch gives today's output exactly.
  - Full-episode run (Clannad E20): at the three T14 spots, median fall in
    0.5 s ≤ 14 dB and near-silent time ≤ 5 %; opening (2:09–3:19) and
    ending (22:20–23:35) songs within 3 dB of today's level; run-time cost
    recorded.
  - You listen to the three spots and both songs: background sounds
    natural, and no music is recognisable.

### T10 — Documentation `[easy]` — done
- Goal: a `README.md` with install and usage, and `CONTRIBUTING.md`
  rewritten for this project (the pasted one describes another project,
  fragdirector).
- Files: `README.md`, `CONTRIBUTING.md`.
- Verify: following the README on this machine works.

## Stage 2 — Fill the gaps the music leaves (approved 2026-11-15)

Goal: in scenes where music was the only background, removing it leaves an
unnaturally empty sound. Stage 2 fills those gaps with fitting background
sound (room tone, wind, crowd, street, nature) so the episode still feels
alive — without bringing music back.

Approach, cheapest and safest first:

1. **Find the gaps.** Places where the removed music was the main
   non-speech sound and the kept effects/ambience are very quiet.
2. **Reuse the episode's own ambience.** For each gap, take quiet,
   speech-free background sound from the *same scene* (scene cuts detected
   from the video), and extend it into the gap with crossfaded, varied
   repeats. It sounds like the show itself and needs no new large model.
3. **Optionally, generate it.** Video-to-audio ("Foley") models can create
   sound from the picture. The well-known ones need 8–20 GB of GPU memory;
   whether any usable one runs on a 4 GB card is unknown and must be measured
   before anything is built on it.
4. **Optionally, a sound library.** You supply a folder of ambience
   recordings (e.g. CC0 sounds); a sound-tagging model (e.g. PANNs, 527
   AudioSet classes) picks a matching one for each gap.

Order: T15 → T16 → T19 give a complete, local feature. T17 and T18 are
independent experiments and go ahead only if you want them.

> **Proposed change after measuring (T15, 2026-11-15) — please confirm.**
> In Clannad E20 the music is audible 74 % of the time and is the main
> background about 60 % of the time (~14 min of gaps). Under the music the
> effects stem is nearly silent (median −77 dB): the studio used music
> *instead of* ambience. The whole episode has only ~28 s of speech-free,
> music-free ambience, and 17 of its 25 minutes have none. So:
> - T16 (reuse the episode's own ambience) can fill only a few gaps; it
>   becomes a small extra layer, not the main path.
> - The main path becomes T18 (a library of ambience recordings), with the
>   scene recognised from the **picture** (an image model such as CLIP)
>   instead of from the sound, since there is no background sound to tag.
> - T17 (generation) stays an experiment.
> - New order: T15 → T18 → T16 → T19; T17 if you want it.
>   (T16 later dropped after measuring, 2026-11-16.)
> - This needs your answer to open question 1 (adding sound that was not
>   in the episode), and a source of ambience recordings (question 4).

### T15 — Find the gaps `[hard]` — done
- Goal: list every stretch (≥ 2 s) where, after music removal, the
  non-speech background is much quieter than it was with music (thresholds
  measured, not guessed), plus video scene cuts (`ffmpeg` scene detection).
- Files: `src/musicremover/gaps.py` (pure numpy on stems + cut list),
  `probe.py` or a new `scenes.py` for the cut detection, tests;
  `experiments/` script to print the gap list for the test episode.
- Verify: `pytest`; the gap list for Clannad E20, and you listen to three of
  them to confirm they feel empty.
- Measured thresholds (0.1 s levels, 1 s smoothing): music audible = music
  stem > −50 dB; a gap = music audible and ≥ 15 dB above the effects stem
  (61 % of the episode; the result barely changes between 6 and 20 dB).
- Built: `gaps.py` (`find_gaps`, `split_at_cuts`, `frame_levels`),
  `scenes.py` (`scene_cuts`, ffmpeg scene score 0.3 on frames scaled to
  256 px, ~17× real time), `tests/test_gaps.py` (8 tests), experiment
  `experiments/list_gaps.py`. 137 tests pass.
- Clannad E20: 294 scene cuts; 63 gaps, 14.6 min, 180 pieces after
  splitting at cuts; list in `output/t15/gaps.csv`. Most gaps are dialogue
  under quiet music (music ≈ −38 dB, background −60 to −100 dB). The opening
  (2:08–3:18) and ending (22:19–23:36) are gaps too — T18 should probably
  not fill songs over credits (new item for T18). The gap at 20:30 is the
  laugh, from stems made before T13; the real pipeline finds gaps after the
  voice rescue.
- Your check (2026-11-16): the sample gaps "kinda do" feel empty. Listened to: `output/t15/1_no_talk_22-19.8.mkv` (the
  ending), `2_talk_13-30.2.mkv` (a minute of dialogue under music) and
  `3_typical_20-51.4.mkv`, and say whether they feel empty.

### T16 — Ambience fill from the episode itself `[hard]` — dropped
- Goal: fill each gap with background sound taken from speech-free, music-
  free moments of the same scene (effects stem), extended by crossfaded,
  randomly offset repeats, at a level matched to that scene's own
  background. Never use material containing speech or music; leave a gap
  empty if the scene has no usable ambience.
- Files: `src/musicremover/fill.py`, tests, CLI option `--fill ambience`.
- Verify: `pytest` (fill never contains speech-stem energy, levels and
  joins are smooth, gaps without source material stay empty); you watch
  the test episode with and without fill.
- Measured first (2026-11-16, `experiments/ambience_sources.py`, T1 scan
  stems): "source" = no speech (< −60 dB), no music (< −50 dB), some
  background (effects > −70 dB), 0.5 s away from speech or music, runs of
  1 s or more. Clannad E20 has **18 s** of it in 10 runs of 1–3.5 s, mostly
  very quiet (−55 to −65 dB); the one loud run (8:37, −36 dB) is probably a
  sound effect, not ambience. With looser limits (effects > −75, runs of
  0.5 s, 0.3 s away): 33 s in 26 runs.
- Of the 180 gap pieces (14.6 min), the number with a source in the **same
  shot**: 0 (strict) or 5 pieces / 35 s (loose). Shots here are short
  (294 cuts, ~5 s each), so "same scene" by cut detection is far stricter
  than a story scene. A source within 30 s: 22 pieces / 95 s (strict) or
  56 / 227 s (loose); within 60 s: 49 / 218 s or 113 / 466 s.
- So, as specified, T16 fills almost nothing (0–35 s of 876 s), and what it
  could fill would be 1–3 s clips looped across gaps of 5–60 s, which tends
  to sound repetitive. Options were:
  1. Drop T16; the library fill (T18b) stays the only fill. Recommended.
  2. Borrow from up to 30 s away, but only from a shot whose SigLIP label
     matches the gap's (reuses T18a), with the loose limits: at most
     ~227 s of the 876 s, short looped sources, risk of wrong-place sound.
  3. Build it exactly as specified (same shot only): almost no effect.
- **Dropped (your decision, 2026-11-16):** the library fill (T18b) stays
  the only fill. No code was written; the measuring script stays in
  `experiments/`.

### T17 — Generated background sound: feasibility on 4 GB `[hard]` — done (waiting for your go/no-go)
- Goal: measure whether a video-to-audio model can generate usable
  background sound for gap scenes on this machine: candidates to check
  include MMAudio (small variants), HunyuanVideo-Foley XL with CPU offload,
  and newer releases. Record GPU memory, time per 10 s, licence, and
  whether the output contains music (it must not).
- Files: `experiments/` only, `docs/generation-choice.md`.
- Verify: three gap scenes generated (or a written reason why not); you
  listen and decide go / no-go.
- Phase 1 done (2026-11-16, Opus agent, no GPU): research in
  `docs/generation-choice.md`. Shortlist: (1) Stable Audio 3 Small SFX
  (text-to-audio from the scene label; code MIT, weights Stability
  Community License + Gemma Terms for its text encoder; 3.45 GB; ~2 GB
  VRAM), (2) HunyuanVideo-Foley XL, community fp8 weights with block swap
  (Tencent Hunyuan Community License, which excludes the EU, UK and South
  Korea; claimed < 4 GB, unproven). Rejected: MMAudio (non-commercial /
  research-only parts), Hunyuan XXL (12–20 GB), ThinkSound/PrismAudio
  (research only), Kling-Foley (no weights), joint audio-video models.
- Downloaded to **D:\music-remover-t17** (11 GB; C: was short), weights
  hash-checked. Stable Audio 3 came from an ungated copy whose files match
  the official (login-gated) repo byte for byte. Extra venv
  `experiments/t17-venv` (214 MB) for Hunyuan; project `.venv` unchanged.
- Scenes (`output/t17/scenes/`): 1:09.4 night_street, 14:12.2 bedroom,
  20:51.4 park (same as the T15/T18b "typical" clip). Talk-free gaps in the
  episode body are rare; the clean ones were credits.
- Before Phase 2 (GPU): you confirm the licences (Stability Community
  License, Gemma Terms, Tencent Hunyuan licence — are you outside the EU,
  UK and South Korea?). Phase 2 runs after T18b.
- Your answer (2026-11-16): "Yes, accept both" — both licences accepted,
  and you are outside the EU, UK and South Korea.
- Phase 2 done (2026-11-16, Opus agent; details in
  `docs/generation-choice.md`):
  - Stable Audio 3 Small SFX (fp16, 8 steps): 1.8 GB VRAM allocated (2.75
    reserved), 6.1 GB RAM, ~1 s per 10 s of audio, load 17 s. Music check
    (Bandit music stem of the generated sound): −114 to −120 dB, 70–84 dB
    below the sound; no speech either. Fits and is clean.
  - HunyuanVideo-Foley XL (fp8, block swap 30/36): 9.5–11 GB VRAM reserved,
    only finished because Windows spilled into RAM; 116–218 s per 10 s;
    voice-like sound where characters are on screen, and music up to
    −40 dB in 2 of 3 scenes. **No-go.**
- Listening clips: `output/t17/sa3/` and `output/t17/hunyuan/`
  (`1_night_street`, `2_bedroom`, `3_park`; track 1 = no-music sound with
  the generated bed at the T18b level, track 2 = generated sound alone);
  compare with `output/t17/scenes/` and, for the park, the library fill.
- Waiting for: you to listen to the Stable Audio 3 clips and say go/no-go.
  If go, a new task adds `--fill generate` (scene label → prompt → short
  generated bed, looped, same level and fades as the library fill; ~1–2
  min per episode).

### T18 — Ambience from a sound library (main path) — split 2026-11-16
Overall goal: `--fill library --library DIR`: recognise each gap scene from
its video frames and lay a matching ambience recording from the library
under the gap, quietly, with fades at scene cuts. The library is a folder of
recordings named or tagged by scene type; nothing with music in it. Split
into two tasks that can each be checked on their own (agreed 2026-11-16):

### T18a — Scene labels from the picture `[hard]` — done
- Goal: for each gap piece, grab a few stills from the video and label the
  scene with an image–text model comparing them with short descriptions
  ("classroom", "street at night", "quiet bedroom", …). Model: Google
  SigLIP base (`google/siglip-base-patch16-224`, Apache-2.0, chosen with
  you 2026-11-16); check licence, download size, GPU memory and speed on
  this laptop first.
- Files: `src/musicremover/scenes.py` (frame grabbing, labelling with an
  injectable model), `pyproject.toml` (`transformers`), tests,
  `experiments/label_gaps.py` (labels + contact sheet for the test episode).
- Verify: `pytest` (fake model, no download); measured memory/time
  recorded here; you look at the contact sheet of gap stills with their
  labels and say whether they are mostly right.
- Built (2026-11-16): `scenes.py` gains `still_times`, `grab_frame`
  (ffmpeg exact seek, 320 px wide RGB), `label_scenes` (one model call for
  all stills; logits averaged over a piece's 3 stills; softmax share as
  score, plus runner-up), `to_pixels` and `SiglipScenes` (pinned revision
  `7fd15f0`, safetensors only, tokenizer read with the `tokenizers` library
  so no sentencepiece/protobuf). `pyproject.toml`: `transformers`.
  `tests/test_scenes.py` (8 tests); 145 tests pass.
  `experiments/label_gaps.py` writes `output/t18a/labels.csv` and the
  contact sheet `output/t18a/index.html`.
- Measured: licence Apache-2.0; download 813 MB (779 MB in the HF cache);
  fp16 on the RTX 3050: 402 MB loaded, 672 MB peak at batch 64, ~220
  stills/s (CPU: ~7/s); load 11 s. Labelling 540 stills: 3.3 s. Grabbing
  them one ffmpeg seek each: 186 s (0.34 s/still), now the slow part.
- Label set tuned on Clannad E20 (checked 60 pieces by eye): with weather
  choices, "rain" took 54 of 180 pieces, nearly all character close-ups.
  Dropping rain/snow and adding `close_up` ("place not visible") fixed
  that: 74 close-ups, school/street/park/shop scenes mostly right.
  Merging home rooms into one "home" choice made things worse (went to
  shop/hospital), so not kept.
- Known limits, for T18b: rooms in a home are often labelled shop, kitchen
  or bedroom interchangeably; about 40 % of pieces are `close_up` and need
  a label from neighbouring pieces of the same scene.
- Your check (2026-11-16): first unsure, then "somehow good, but there are
  inaccuracies". Good enough to build T18b on; the real test is listening.
  If wrong scenes are audible, a later task can tune the label set or let
  the library group similar scenes.

### T18b — Lay library ambience into the gaps `[hard]` — in progress
- Goal: library folder format (sub-folder per scene type), pick a recording
  per gap piece by scene label, lay it under the gap quietly with fades at
  scene cuts; `--fill library --library DIR`.
- Files: `src/musicremover/library.py`, `cli.py` (minimal flag; T19 does
  the full option set), tests.
- Verify: `pytest` with a tiny generated library; by hand on the episode.
- Opening and ending songs (decided 2026-11-16): never fill inside `--keep`
  ranges; also skip any gap of 60 s or more with under 20 % talking that
  lies in the first or last 5 minutes, and print each one skipped.
- Needs some real CC0 ambience recordings for the by-hand check (the
  download list for the docs is T19).
- Built (2026-11-16): `library.py` — `load_library` (sub-folder per scene;
  `close_up` reserved), `split_songs` (the agreed opening/ending rule),
  `subtract_ranges` (`--keep`), `plan_fill` (close-ups borrow the nearest
  real label in the same gap, else from a gap ≤ 30 s away, else stay
  empty; touching same-scene pieces become one placement), `render_fill`
  (random excerpt, short files looped with 2 s crossfades, level = removed
  music − 12 dB kept in −55…−40 dBFS, 1 s fades at gap ends, 0.5 s
  equal-power crossfade at scene changes, seeded so runs repeat),
  `make_fill` (ties it to the video). `scenes.grab_frames` grabs 4 stills
  at once. `cli.py`: `--fill {none,library}` and `--library DIR`; the fill
  is added to the effects stem before mixing, so the limiter still applies;
  SigLIP is loaded only when there is something to fill. Tests:
  `tests/test_library.py` (16) and 4 new CLI tests; 164 pass.
- Episode run with a stand-in library of soft noise (no real recordings
  yet; `output/t18b/noise-library`): 9:13 for the whole episode on the
  RTX 3050, no out-of-memory with both models. Skipped as songs: 2:09–3:19
  and 22:20–23:35 (both right). Ambience under 11:58 in 86 places.
  Clips for loudness/fades: `output/t18b/2_talk_filled.mkv`,
  `3_typical_filled.mkv` (same times as the T15 clips).
- Left to verify: listen with real CC0 recordings in a few scene folders.
- Episode run with the real library (2026-11-16, after T20,
  `output/t18b/real/`): 10:25, songs skipped as before, ambience under
  11:55 in 83 places. Listening clips in `output/t18b/listen/`, each spot
  as `_0_original`, `_1_no-fill` (T20 run, default options) and
  `_2_with-fill`: night street 15:44, schoolyard 11:52, forest 1:49,
  home/shop 4:51, bedroom 16:33, talk 13:30 (T15 clip), beach 14:36. Fill
  alone measures −43 to −53 dB under clip levels of −26 to −39 dB, so it
  sits well under the talk.
- Waiting for: you to listen and say (1) whether the fill makes the scenes
  feel less empty, (2) whether any spot sounds wrong for the picture or too
  loud/quiet (`--fill-level` moves it), (3) whether the fill should be on
  by default (T19 / open question 3).
- CC0 sources found (2026-11-16, for this check and T19's list):
  - **BigSoundBank** (bigsoundbank.com, Joseph Sardin): sounds marked
    "Free and Royalty Free" are CC0; no account, no attribution needed;
    WAV/FLAC/OGG. 267 ambiences; sub-categories by URL
    `/search?CatID=` + `AMBSchl` (school), `AMBUrbn` (urban), `AMBRurl`
    (rural), `AMBForst` (forest), `AMBSea` (seaside), `AMBRest`
    (restaurant & bar), `AMBRoom` (room tone), `AMBMrkt` (market); also
    `TRN` (trains), `RAIN`, `BIRD`. Mostly recorded in France: voices are
    French, so prefer takes without clear talking. Main source.
  - **Freesound** (freesound.org) with the licence filter "Creative
    Commons 0": far more, including recordings made in Japan (stations,
    cicadas, streets); a free account is needed to download.
  - Not CC0, so left out: Sonniss GDC bundles (royalty-free, no
    attribution, but their own licence), itch.io/Gumroad free packs
    (licences vary).
  - Watch for music in restaurant, market and shop recordings.
- Starter set for the check (folder ← BigSoundBank category), matching the
  most common labels in Clannad E20: `schoolyard` and `classroom` ←
  School; `night_street` ← Rural/Urban at night (e.g. "Campaign at
  Night"); `shop` ← Restaurant & Bar (no music); `living_room`,
  `bedroom`, `kitchen` ← Room Tone; `forest` ← Forest; `beach` ←
  Seaside; `park` ← Birds/Rural; `city_street` ← Urban. One or two files
  each is enough; scenes without a folder are labelled from the others.
- Starter library downloaded (2026-11-16, by Claude at your request) into
  `samples/ambience/` (ignored by Git), 438 MB, all BigSoundBank pages
  checked to say "CC0 (public domain): Free and royalty-free". Folder ←
  sound ID: night_street ← 1469, 1470 (night insects, wind); schoolyard ←
  1072 (playground football); bedroom ← 2387 and kitchen ← 2458 (apartment
  / house room tone), living_room and shop reuse those two (their own picks
  0750/0080 had no direct FLAC); forest ← 2715; beach ← 1447; park ← 0912
  (morning birds, 96 kHz, resampled on load); countryside ← 1911; city_street
  ← 3030 (Paris crossing); quiet_street ← 1713 (Paris street). Left out:
  0909 "Bus stop in Japan" (ads may contain music); 0910 and 0911 (direct
  download refused). No classroom/school_corridor folder: those shots take
  the nearest other label. Recorded in France: watch for clear French talk.
- From T18a: give `close_up` pieces the label of the nearest labelled
  piece in the same gap (or leave them to the gap's majority label); treat
  home rooms and shop as one quiet-indoor sound unless the library has
  better; grabbing stills costs ~3 min per episode (one seek each) — try
  fewer stills or one decode pass; load SigLIP after the separator has
  freed the GPU.

### T19 — Fill options and docs `[medium]` — done
- Goal: `--fill {none,ambience,library,generate}` (default decided after
  you listen), `--fill-level DB`, README section.
- Files: `cli.py`, `README.md`, tests.
- Verify: `pytest`; README examples run.
- Done (2026-11-16): `--fill-level DB` added (an offset on the fill's
  level: default 0, allowed −20…20, only with `--fill library`; passed
  through `make_fill` → `render_fill`). README section "Filling the gaps the
  music leaves": how it works, library folder format, scene names, options,
  where to get CC0 recordings. 168 tests pass (4 new); `--help` and the
  option-error messages checked by hand. The README's full-run examples were
  not run on a real episode (no real recordings in the library yet).
- **Changed from the goal (please confirm):** `--fill` has only `none` and
  `library` for now. `ambience` (T16, dropped) and `generate` (T17) are not built, so
  offering them would be a flag that does nothing; add each choice when its
  task lands. The default stays `none` until you have listened with real
  recordings (T18b "left to verify").

### Stage 2 answers (2026-11-15)
- Adding sound that was not in the episode: **yes**. The proposed change
  above (library as main path, scene from the picture) is confirmed.
- Ambience source: **a CC0 pack the user downloads by hand** from a list the
  tool's docs provide; no accounts or API keys.
- Generation (T17): **laptop only**.
- Fill on by default: still open (decide after listening).

### Stage 2 open questions
1. Is it OK for the tool to *add* sound that was not in the episode
   (generated or from a library), or should it only reuse the episode's own
   sound?
2. Must everything run locally on this laptop, or is a cloud GPU acceptable
   for generation (T17)?
3. Should the fill be on by default once it works?
4. Where should ambience recordings come from? Options: you collect them;
   or the tool downloads a fixed set of CC0 recordings (e.g. from Freesound,
   which needs a free account and API key that *you* set up); or a curated
   pack you download by hand.

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
