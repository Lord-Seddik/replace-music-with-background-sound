# Model choice (task T1)

Status: **decided 2026-11-15: Bandit v2 multilingual, confirmed by listening.**

Test episode: Clannad S01E20 (1080p BluRay, Japanese stereo Opus track), 24.2 min.
Machine: RTX 3050 Laptop 4 GB, PyTorch 2.11 + CUDA 12.8. Date: 2026-11-15.

## Recommendation

**Bandit v2, multilingual checkpoint** (`checkpoint-multi.ckpt`).

- It removed the opening song best, singing included (see clip 4), so the
  plan's "sung vocals may stay" limit is smaller than feared.
- It is trained on speech in more than 30 languages, which suits Japanese
  dialogue. BandIt Plus was trained on English only.
- It fits easily in 4 GB (1.3 GB peak) and separates a full episode in about
  17 minutes with the faster settings below.

This is a recommendation from measurements. The decision is yours, after
listening.

## How to listen

Open the files in `output/bakeoff/watch/`. Each clip comes as:

| File | What you hear |
|---|---|
| `clipN_0_original.mkv` | the original |
| `clipN_1_bandit_v2_multi.mkv` | recommended model: speech + effects |
| `clipN_2_bandit_plus.mkv` | the other cinematic model |
| `clipN_3_demucs_baseline.mkv` | ordinary music model (voice only, effects lost) |

| Clip | Starts at | What it tests | Listen for |
|---|---|---|---|
| 1 | 6:03 | talk over quiet music | music fully gone? voices natural? |
| 2 | 12:06 | talk over quiet music | same |
| 3 | 18:09 | talk, almost no music | nothing should be lost or damaged |
| 4 | 2:15 | opening song, no talk | should be near silence; any singing left? |
| 5 | 20:59 | talk + music + effects (busiest scene) | effects kept, music gone, no "underwater" sound |

Clips 1–3 were cut at fixed points (25/50/75 %). Clips 4–5 were picked from
a full-episode scan of the loudest music and the busiest effects.

Separate stems (`speech.wav`, `music.wav`, `effects.wav`) are in
`output/bakeoff/<model>/<clip>/` if you want to hear what was removed.

## Measurements

Loudness of the "no music" result compared with the original clip (dB RMS):

| Clip | Original | Bandit v2 multi | BandIt Plus | Demucs |
|---|---|---|---|---|
| 1 | −29.0 | speech −29.7, music −40.1 removed | music −39.6 removed | music-like −42 removed |
| 4 (opening song) | −24.4 | **−52.3** (28 dB removed) | −42.9 (18 dB) | −27.6 (3 dB: singing kept) |
| 5 (busy scene) | −27.0 | −28.4 (effects −33.7) | −28.2 (effects −31.6) | −27.9 (effects not separable) |

All models' stems add back up to the original within −61 dB or better, so
nothing is lost in the splitting itself.

Speed and GPU memory per one-minute clip, default settings:

| Model | Speed (× real time) | Peak GPU memory |
|---|---|---|
| Bandit v2 multi | 0.4–0.9 | 1.3 GB |
| BandIt Plus | 0.5–0.7 | 0.9 GB |
| Demucs (baseline) | 2.8–7 | 0.55 GB |

Bandit v2 settings tried on clip 1:

| Overlap | Batch | Speed | Peak memory | Note |
|---|---|---|---|---|
| 4 | 1 | 0.68× | 1.3 GB | default quality |
| 2 | 1 | **1.59×** | 1.3 GB | speech differs from overlap 4 by −27 dB (very small) |
| 2 or 4 | 2–4 | 0.25× | 2.3–4.3 GB | slower: the card spills into shared memory |
| 2 | 8 | — | out of memory | |

Full episode, overlap 2, batch 1, fed in 5-minute pieces: **24.2 min in
17.5 min (1.38× real time)**.

## What T1 found that later tasks need

- **Long input must be fed in pieces.** The runner keeps a whole input and
  all its stems on the GPU; a full episode in one go crashed with a cuDNN
  error. T4 already plans chunking; the pieces need overlap and a crossfade
  so the joins are seamless (the scan used plain 5-minute cuts, which is fine
  only for measuring).
- **Use overlap 2, batch 1** as the default; overlap 4 can be a `--quality`
  option.
- **Dependencies:** the runner needed `pytorch_lightning` for Bandit v2, and
  `asteroid`, `spafe`, `pedalboard`, `torch_audiomentations` etc. only for
  BandIt Plus. Choosing Bandit v2 avoids most of them. T4 should copy the
  few Bandit v2 model files into the project (MIT licence) instead of
  importing the whole cloned repo.
- **Licences:** the runner code is MIT. The Bandit v2 weights are
  **CC-BY-SA 4.0**: fine for personal use; if the tool is ever shared, the
  weights should be downloaded by the user rather than bundled, with credit.
- **Disk:** `output/bakeoff/` is 3.9 GB, mostly the full-episode scan stems
  in `output/bakeoff/scan/`. They can be deleted after you have listened.

## Sources

- Bandit v2 paper: Watcharasupat, Wu, Orife, "Remastering Divide and Remaster", https://arxiv.org/abs/2407.07275
- Bandit v2 weights: https://zenodo.org/records/12701995
- Runner: https://github.com/ZFTurbo/Music-Source-Separation-Training (commit 1f700c3)
