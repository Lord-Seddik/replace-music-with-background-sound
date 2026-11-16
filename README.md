# music-remover

Removes the background music from a video's soundtrack, keeping speech and
sound effects, and writes a new video next to it. Made for anime episodes.

```
music-remover "Episode 01.mkv"      ->  Episode 01.nomusic.mkv
```

The original file is never changed. Video, subtitles, fonts and chapters are
copied without re-encoding; only the sound is replaced.

## Requirements

- Windows or Linux (so far it has only been run on Windows)
- Python 3.12
- FFmpeg (with ffprobe) on your PATH
- An NVIDIA GPU is recommended. It also works on the CPU, but that is very
  slow. Measured on an RTX 3050 Laptop (4 GB): the GPU runs at 4.3 times
  real time, so a 24-minute episode takes about 6 minutes; the CPU runs at
  0.14 times real time, so the same episode takes about 3 hours.

## Install

Create a virtual environment, then install PyTorch with CUDA first:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

For a CPU-only install, leave out the `--index-url` option.

### Model weights

The separation model's weights are not included. Download
`checkpoint-multi.ckpt` from
https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt and put it in
one of these places:

- `models/bandit_v2_multi.ckpt` (inside the project folder), or
- `~/.cache/music-remover/checkpoint-multi.ckpt`

Or point to it yourself with `--weights PATH` or the `MUSICREMOVER_WEIGHTS`
environment variable. The lookup order is: `--weights`, then
`MUSICREMOVER_WEIGHTS`, then `models/bandit_v2_multi.ckpt`, then the cache
folder.

The weights are licensed CC-BY-SA 4.0. They are not part of this project and
must be downloaded by you.

## Usage

```bash
music-remover "Episode 01.mkv"
music-remover "D:\Anime\Season 1"          # a folder: every video in it
```

Outputs are written next to each input as `NAME.nomusic.EXT`. Inputs that
already have an output are skipped.

Main options (`music-remover --help` lists them all):

- `-o`, `--out-dir DIR`: write the outputs to this folder instead.
- `--language jpn`: use the audio track in this language (default: the
  default track).
- `--music-gain -20`: keep the music 20 dB quieter instead of removing it
  completely. This hides artefacts where the model struggles. Must be 0 or
  negative.
- `--keep 0:00-1:30 --keep 22:10-end`: leave these parts of the sound
  untouched, for example the opening and ending. Repeat the option for more
  ranges; `end` means the end of the episode.
- `--keep-original`: also keep the original audio track in the output file.
- `--quality high`: a slower setting that gives slightly better results
  (about twice as slow as the default).
- `--device cpu`: force the CPU (`auto` is the default and uses the GPU if
  there is one; `cuda` forces the GPU).
- `--weights PATH`: use a weights file from another place.

## Filling the gaps the music leaves (optional)

In many scenes the music was the only background sound, so with it gone the
scene can feel unnaturally empty. `--fill library` lays a quiet, music-free
ambience recording under those stretches. It is off by default
(`--fill none`).

```bash
music-remover "Episode 01.mkv" --fill library --library "D:\Ambience"
music-remover "Episode 01.mkv" --fill library --library "D:\Ambience" --fill-level -3
```

How it works: the tool finds the stretches where removed music was the main
background, looks at a few frames of the picture to tell what kind of scene
it is (classroom, street at night, park, ...), and picks a recording of that
kind from your library. The recording fades in and out at the edges of the
stretch and crossfades where the scene changes. It never fills anything
covered by `--keep`, nor a long, mostly wordless stretch in the first or last
five minutes (the opening and ending songs); each one skipped is printed.
Running twice gives the same result. This adds about 3 minutes per episode
and loads a second, small model (SigLIP, 813 MB download on first use).

**The library is a folder with one sub-folder per scene type**, holding
music-free recordings (WAV, FLAC, OGG or MP3; at least 1 second each, and
short ones are looped to fit):

```
Ambience/
  classroom/    room-tone.wav
  schoolyard/   school-break.flac
  night_street/ night-1.wav, night-2.wav
  park/         birds.wav
```

Only the scene types you have a folder for are ever chosen, so start with a
few. These folder names have a tuned description for the picture model:
`classroom`, `school_corridor`, `schoolyard`, `city_street`, `quiet_street`,
`night_street`, `park`, `countryside`, `forest`, `beach`, `living_room`,
`bedroom`, `kitchen`, `hospital`, `shop`, `train` and `workplace`. Any other
folder name also works; it is described to the model as "a <name>" (for
example `rooftop`). `close_up` is reserved: shots where the place is not
visible take the scene of the nearest shot that shows it.

Options:

- `--fill library`: turn the fill on. Needs `--library`.
- `--library DIR`: the folder described above.
- `--fill-level DB`: change the loudness of the fill. By default it is 12 dB
  below the music it replaces, kept between -55 and -40 dBFS. `-3` makes it
  3 dB quieter, `3` makes it 3 dB louder. Allowed: -20 to 20.

### Where to get recordings

Use recordings that are free to use without attribution (CC0), and check
that they contain no music or clear talking:

- **BigSoundBank** (bigsoundbank.com): sounds marked "Free and Royalty Free"
  are CC0; no account needed. Useful categories: School (`classroom`,
  `schoolyard`), Urban and Rural (`city_street`, `night_street`, `park`),
  Room Tone (`living_room`, `bedroom`, `kitchen`), Forest, Seaside,
  Restaurant & Bar (`shop`). Mostly recorded in France, so voices in the
  background are French.
- **Freesound** (freesound.org): use the licence filter "Creative Commons 0".
  Much larger, including recordings made in Japan. A free account is needed
  to download.

## Known limits

- Sung vocals (insert songs, characters singing) may sometimes stay, because
  the model hears them as a voice. In the test the opening song was removed
  well.
- Musical sound effects (a chime, a sting) may be removed together with the
  music.
- 5.1 sound is mixed down to stereo first.

## Credits and licences

- Model: Bandit v2 by Watcharasupat, Wu and Orife, "Remastering Divide and
  Remaster" (https://arxiv.org/abs/2407.07275).
- Weights: `checkpoint-multi.ckpt`, CC-BY-SA 4.0, from Zenodo
  (https://zenodo.org/records/12701995). Not included in this repository.
- Model code: copied from ZFTurbo/Music-Source-Separation-Training (MIT
  licence) into `src/musicremover/separate/bandit_v2/`, which keeps its
  licence notice.
- This project's own licence: not chosen yet (there is no LICENSE file).

Decisions behind the model choice are in `docs/model-choice.md`.
