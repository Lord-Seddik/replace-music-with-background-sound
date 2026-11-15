# Contributing to music-remover

Thanks for looking at this. The project is small; the [README](README.md)
shows how to install it and what it does.

## Setup

Follow the install steps in the [README](README.md) (virtual environment,
PyTorch with CUDA, then the model weights), then:

```bash
pip install -e ".[dev]"
pytest
```

The tests do not need a GPU, the model weights or real episodes. They use
generated media and a fake separator. Real episodes go in `samples/` and
results in `output/`; Git ignores both folders.

## Layout

The code is in `src/musicremover/`. Each stage is a module with a small
interface, so one can be replaced without touching the others:

- `probe.py` is the only module that runs ffprobe; nothing else calls it.
- `extract.py` writes the chosen audio track to WAV.
- `separate/` holds the `Separator` interface and the chunking that feeds it
  (`__init__.py`), the Bandit v2 backend (`bandit.py`), and the copied model
  code (`bandit_v2/`). Keep the licence notice in `bandit_v2/`.
- `mix.py` is pure numpy. It works on arrays and touches no files.
- `mux.py` is the only module that writes video files.
- `cli.py` ties the stages together.

`experiments/` holds the scripts from the model bake-off (task T1). They are
throwaway and not part of the tool.

The plan and the reasons for the design are in [PLAN.md](PLAN.md); the model
choice is in [docs/model-choice.md](docs/model-choice.md).

## Working on a change

- Keep each change to one thing. If you find something else that needs
  fixing, add it to `PLAN.md` as a new task rather than fixing it in the same
  change.
- Add or update tests for the change, then run `pytest`.
- Check the change on a short clip before a full episode. A full run takes
  many minutes even on a GPU.
- Do not commit episodes, output or model weights. `.gitignore` already
  covers `samples/`, `output/`, `models/*.ckpt` and the media files.

## Things to keep in mind

- **FFmpeg** is always run with an argument list built by the code, never
  with a shell string. Keep it that way.
- **Input files** are never modified, and an existing output is never
  overwritten. Check both before writing anything.
- **Checkpoints** are loaded with `torch.load(..., weights_only=True)`. Keep
  that flag, so a checkpoint cannot run code when it is loaded.
- **Weights are not bundled.** They are CC-BY-SA 4.0, so users download them
  themselves (see the README).
- **4 GB GPUs must keep working.** The separator processes one chunk at a
  time with batch size 1. Test any change to the chunking or the model
  settings against that limit.
- Work is planned in `PLAN.md`. See `CLAUDE.md` for the workflow: plan first,
  one task at a time, and the difficulty tag for each task.

## Commit messages

Short imperative subject, for example `Add crossfade option to mix`. In the
description, say what changed, how you checked it, and what you did not
check.

## License

None chosen yet. There is no LICENSE file, so until one is added, the code
is not open for reuse by others.
