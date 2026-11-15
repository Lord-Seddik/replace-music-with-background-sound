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
