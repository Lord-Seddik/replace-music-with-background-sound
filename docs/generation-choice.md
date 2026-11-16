# Generated background sound (task T17)

Status: **measured, 2026-11-16. Three scenes generated with each of two
models. Waiting for you to listen and decide go / no-go.** Recommendation
from the measurements: Stable Audio 3 Small SFX is worth your listen;
HunyuanVideo-Foley XL is no-go (see "Results"). Licences accepted by you
on 2026-11-16 ("Yes, accept both").

Machine: RTX 3050 Laptop 4 GB, 15 GB RAM, PyTorch 2.11 + CUDA 12.8, Windows 11.
Rule from Stage 2: generation runs on this laptop only, no cloud.

## What is needed

For each gap scene, about 5–60 s of quiet background sound (room tone, night
insects, wind, birds, distant street) that fits the picture and **contains no
music**. It does not have to be synchronised to on-screen actions: the
effects stem keeps the episode's own footsteps and doors. That matters
for the choice. A video-to-audio ("Foley") model reads the picture. A
text-to-audio model given the scene label from T18a ("night_street",
"bedroom", …) can also make ambience, and is much smaller.

## Candidates (checked 2026-11-16)

| Model | Kind | Licence (code / weights) | OK for this use? | Download | GPU memory | Verdict |
|---|---|---|---|---|---|---|
| **Stable Audio 3 Small SFX** (Stability AI, May 2026) | text → audio, 433 M DiT + SAME-S decoder, 44.1 kHz stereo, up to 120 s | MIT / Stability AI Community License; text encoder T5Gemma under Gemma Terms of Use | **Yes**: personal use is "Non-Commercial Purpose", free; commercial use free under US $1 M revenue (registration needed) | 3.45 GB (2.27 GB checkpoint + 1.18 GB T5Gemma), safetensors | 1.7–2.4 GB peak (Stability, H200, 5–120 s); also runs on CPU | **Shortlist #1, try first** |
| **HunyuanVideo-Foley XL** (Tencent, 2025) with fp8 weights + block swap | video (+ text) → audio, 48 kHz | Tencent Hunyuan Community License (code and weights) | **Yes, outside the EU, UK and South Korea** (the licence does not apply there); free below 100 M monthly users | official 10.6 GB (.pth); fp8 set used here 6.9 GB: 3.75 GB XL fp8 (community conversion, phazei) + 0.47 GB Synchformer + 0.74 GB DAC-VAE + 1.5 GB SigLIP2 + 0.78 GB CLAP | official 16 GB, 8 GB with `--enable_offload`; ComfyUI node author: fp8 + block swap "under 4 GB", up to ~60 s per 5 s of audio | **Shortlist #2**: the only true video-to-audio candidate that might fit |
| MMAudio small_44k / large_44k_v2 (2024–25) | video (+ text) → audio | MIT / **CC BY-NC 4.0**, and it needs Apple **DFN5B-CLIP**, licensed for **"Research Purposes" only** (excludes personal or product use) | **No** (the CLIP part forbids use outside research) | small: 0.63 GB + 1.2 GB VAE + 0.95 GB Synchformer + 3.9 GB CLIP + 0.49 GB BigVGAN ≈ 7.2 GB (≈ 4.2 GB with fp16 repacks) | large v2: ~6 GB at 16-bit (README); small: not published, probably 3–4 GB (CLIP-H alone ~2 GB fp16) | Rejected on licence; not downloaded. Its training set (VGGSound) also includes music classes |
| HunyuanVideo-Foley XXL | video → audio | as XL | as XL | 10.3 GB main model alone | 20 GB, 12 GB with offload | Rejected: too big |
| ThinkSound / PrismAudio (Alibaba FunAudioLLM, 2025–26) | video → audio with reasoning | Apache-2.0 or MIT code, but code, models and data "for research and educational purposes only" | **No** | PrismAudio ~5.8 GB plus T5-Gemma (gated), VideoPrism, Synchformer | ~5.6 GB (paper, A800) | Rejected: licence and memory |
| ControlFoley (Xiaomi, 2026) | video → audio | Apache-2.0 / CC BY-NC 4.0 | Maybe (personal use only) | not checked | not checked | Not checked in depth: found late, nothing suggests it is smaller than MMAudio |
| Kling-Foley (Kuaishou, 2025) | video → audio | weights not released (only a benchmark) | n/a | n/a | n/a | Not available |
| LTX-2 / 2.3 (+ Foley LoRA), MOVA, Ovi, NAVA, FastH3 / Pika H3 (2026) | make new video **and** audio together | various | n/a | 7–30+ GB | 8–32 GB even with fp8 | Rejected: wrong job (they make new video) and far too big |
| Stable Audio Open Small 1.0 (2025) | text → audio, 11 s max | Stability AI Community License (gated) | Yes | ~1.4 GB | small | Replaced by Stable Audio 3 Small SFX; not shortlisted |

Shortlist: **two** models. MMAudio would have been the obvious third, but its
CLIP image encoder only allows research use. That blocks using it in the
tool, even privately. If you want it as a quality reference ("experiment only,
never in the tool"), say so and it can be added.

### Things to watch for

- **Music in the output.** Both shortlisted models were trained on data that
  includes music: Stable Audio 3 on AudioSparx and Freesound; Hunyuan on a
  large V2A set. Phase 2 checks every result two ways: you listen, and the
  project's own separator (Bandit v2) measures the music stem of the
  generated audio. The tool calls music audible above −50 dB.
- Stable Audio 3 Small SFX is a "post-trained" (distilled, 8-step) model:
  **negative prompts and CFG do nothing** on it. The prompt has to say what
  to make ("field recording, no music"). Hunyuan does take a negative prompt
  (`music, melody, singing, …`).
- The Stable Audio 3 weights were fetched from the ungated mirror
  `cocktailpeanut/stable-audio-3-small-sfx`, because the official repo needs a
  Hugging Face account. Every weight and tokenizer file was checked to have
  the same SHA-256 as the official `stabilityai/stable-audio-3-small-sfx`, and
  the licence files are identical. Only `model_config.json` differs (it names
  the mirror repo); the script overrides that entry and reads the text encoder
  from disk. **By using the model you accept the Stability AI Community
  License and the Gemma Terms of Use**: please read them
  (`D:\music-remover-t17\weights\sa3-small-sfx\LICENSE.md`, `LICENSE_GEMMA.md`).
- The Hunyuan fp8 file is a community conversion (phazei) of Tencent's
  weights. Hashes match phazei's repo, but there is no official fp8 file to
  compare with. Using it means accepting the Tencent Hunyuan Community
  License, which does not apply in the EU, UK or South Korea.

## The three gap scenes

Picked from `output/t15/gaps.csv` and `output/t18a/labels.csv` with
`experiments/t17_pick_scenes.py` plus a look at the frames. Clannad E20 has
almost no talk-free gaps in its body. The only 5–15 s talk-free ones with a
single label were inside the opening (forest, 1:50; it is the opening
credits) and the ending (schoolyard and beach, 21:59–22:19; ending credits).
So two scenes have some talk; the generated sound goes under it, as it
would in real use.

| Clip (`output/t17/scenes/`) | Episode time | Length | Label | Talking | Removed music |
|---|---|---|---|---|---|
| `1_night_street_01-09.4.mkv` | 1:09.4–1:14.6 | 5.2 s | night_street (outdoors at night, street lamp, trees) | 15 % | −39.7 dB |
| `2_bedroom_14-12.2.mkv` | 14:12.2–14:24.2 | 12.0 s | bedroom (tatami room, futon, daytime) | 38 % | −39.1 dB |
| `3_park_20-51.4.mkv` | 20:51.4–20:58.7 | 7.3 s | park (outdoors, daytime, trees) | 56 % | −41.2 dB |

Each clip is 480p H.264 with two audio tracks: the music-free sound (T9) and
the original. Scene 3 is at the same time as T15's `3_typical_20-51.4.mkv` and
T18b's `3_typical_filled.mkv`, so library fill and generated fill can be
compared there.

## Setup done in phase 1 (no GPU used)

- Downloads, all on **D:** in a new folder `D:\music-remover-t17\` (C: has
  little space):
  - `src\stable-audio-3` (Stability-AI/stable-audio-3 @ 790793b),
    `src\ComfyUI-HunyuanVideo-Foley` (phazei @ afd2960), `src\MMAudio`
    (@ 974010a, code only, read for the licence check; not used).
  - `weights\sa3-small-sfx` (3.3 GB), `weights\hunyuan-foley` (4.7 GB),
    `hf\` Hugging Face cache with SigLIP2 and CLAP (2.2 GB). All files
    hash-checked against the Hub.
- `experiments/t17-venv/` (214 MB): a separate venv for the Hunyuan node.
  It reads the project's `.venv` packages through a `.pth` file and adds only
  torchvision 0.26.0+cu128 (matching torch 2.11), diffusers, accelerate,
  timm, av, loguru, sentencepiece and pillow. **The project's `.venv` is
  unchanged** (torch 2.11.0, torchaudio 2.11.0, transformers 5.23.0,
  numpy 2.5.4). Stable Audio 3 needs nothing new: it runs with the project
  Python (its `torch==2.7.1` pin is not needed; loading was checked on CPU).
- Scripts (throwaway, `experiments/`):
  - `t17_pick_scenes.py`: finds the talk-free, single-label stretches.
  - `t17_common.py`: scene table and prompts, the T18b level rule (removed
    music − 12 dB, kept in −55…−40 dBFS), the Bandit v2 music check, and
    listening clips (no-music track + generated sound with 1 s fades; raw
    generated sound as a second track).
  - `t17_sa3.py`: Stable Audio 3 Small SFX. Loading tested on CPU: 23 s,
    no missing weights.
  - `t17_hunyuan.py` + `t17_comfy_stub/`: runs the ComfyUI node's code
    without ComfyUI. The stub supplies the five ComfyUI functions it calls.
    Import tested in phase 1; full runs in phase 2 (peak RAM 10.6 GB).

## Phase 2 commands (run 2026-11-16)

1. `.venv/Scripts/python.exe experiments/t17_sa3.py --device cuda --half`
   → `output/t17/sa3/`: three scenes plus a 10 s timing run;
   `results_cuda.json` with load time, s per 10 s, peak GPU memory,
   peak RAM, and music check.
2. `experiments/t17-venv/Scripts/python.exe experiments/t17_hunyuan.py --device cuda --swap 30`
   → `output/t17/hunyuan/`. If it runs out of memory: `--swap 36` (all
   blocks in RAM). If there is room: lower `--swap` for speed. Optionally
   `--text-only` to see how much the picture helps.
3. Fill in the results table below, then you listen to
   `output/t17/<model>/<scene>_filled.mkv` and decide go / no-go.

## Results (phase 2, 2026-11-16, RTX 3050 4 GB, nothing else on the GPU)

| Model | Load | Time per 10 s of audio | Peak GPU memory | Peak RAM (process) | Music in output? | Your verdict |
|---|---|---|---|---|---|---|
| **Stable Audio 3 Small SFX**, fp16, 8 steps | 17 s | **~1 s** (0.74–1.23 s; the first call 4.3 s incl. warm-up) | **1.8 GB** allocated, 2.75 GB reserved: fits | 6.1 GB (fp32 load, then halved) | **None found**: music stem −114 to −120 dB, 70–84 dB below the sound itself; speech stem also silent | |
| HunyuanVideo-Foley XL, fp8, block swap 30 / 36 | 28–29 s | **116–218 s** | **4.64 GB** allocated (2.8–4.6 GB), 9.5–11 GB reserved: **does not fit**; it only ran because the Windows driver spills into shared system memory | **10.6 GB** | **Yes in 2 of 3**: music stem 15–17 dB below the sound (loudest second −40 to −45 dB); mostly classed as **speech** | |

Per scene. Music check = the project's Bandit v2 on the generated sound
alone (before it is turned down for the mix); "music − total" is how far
the music stem sits below the whole sound.

| Scene | Model | Length | Generation time | Raw level | Music stem | Loudest 1 s of music | Music − total | Speech stem |
|---|---|---|---|---|---|---|---|---|
| 1 night_street | SA3 | 5.2 s | 2.2 s (warm-up) | −30.1 dB | −113.9 | −111.7 | −83.9 | −119.1 |
| 2 bedroom | SA3 | 12.0 s | 0.9 s | −48.1 dB | −117.8 | −114.6 | −69.7 | −102.0 |
| 3 park | SA3 | 7.3 s | 0.9 s | −41.7 dB | −119.6 | −119.5 | −77.9 | −119.9 |
| 1 night_street | Hunyuan | 5.2 s | 103–113 s | −35.7 dB | −119.7 | −118.8 | −84.0 | **−36.0** |
| 2 bedroom | Hunyuan | 12.0 s | 139–152 s | −38.9 dB | **−55.5** | **−44.7** | **−16.6** | **−39.4** |
| 3 park | Hunyuan | 7.3 s | 119–131 s | −28.8 dB | **−44.1** | **−39.7** | **−15.3** | **−29.0** |

Findings:
- **Stable Audio 3 Small SFX** is fast enough for a whole episode: ~14 min
  of gaps ≈ 1–2 min of generation, plus 17 s to load. It fits easily next
  to nothing else on 4 GB; it would have to run after the separator has
  freed the card (like SigLIP in T18b). The separator finds no music and no
  speech in its output. Bedroom and park are steady beds (per-second level
  within ±3 dB). Night street has one louder event in seconds 2–4 (−27 dB
  against −36 to −41 dB): listen for whether it sounds natural.
- **HunyuanVideo-Foley XL** does not fit in 4 GB. The peak is the same
  with 30 or all 36 blocks swapped, so the overflow is outside the
  transformer blocks: feature extraction or the DAC decoder. It runs only
  because Windows lets CUDA spill into system memory, at about 2–3.5 min
  per 10 s (≈ 30–45 min per episode of gaps). Worse, reading the picture
  makes it produce **voice-like sound** wherever characters are on screen:
  Bandit puts almost all of it in the speech stem. In two scenes it also
  added sound that Bandit puts in the music stem. The bedroom output is
  near-silent for 8 s with bursts at both ends. On these measurements it
  is **no-go**, unless your listening says otherwise.

Recommendation: **go for listening with Stable Audio 3 Small SFX, no-go for
Hunyuan.** If the Stable Audio 3 clips sound right, a follow-up task could
add `--fill generate` (scene label → prompt → generated bed, at the same
level and fades as the library fill). It would generate a few seconds per
gap and loop or crossfade them, so a whole episode stays at ~1–2 min.

## How to listen

Each `*_filled.mkv` has the clip's video and two audio tracks: (1) the
music-free sound with the generated bed laid under it at the T18b level
(removed music − 12 dB, −51 to −53 dBFS here, 1 s fades), and (2) the
generated sound alone at full level. The original clips (music-free sound
and original sound) are in `output/t17/scenes/`.

| Scene | Stable Audio 3 | Hunyuan | Compare with |
|---|---|---|---|
| 1 night_street, 1:09.4 | `output/t17/sa3/1_night_street_01-09.4_filled.mkv` | `output/t17/hunyuan/1_night_street_01-09.4_filled.mkv` | `output/t17/scenes/1_night_street_01-09.4.mkv` |
| 2 bedroom, 14:12.2 | `output/t17/sa3/2_bedroom_14-12.2_filled.mkv` | `output/t17/hunyuan/2_bedroom_14-12.2_filled.mkv` | `output/t17/scenes/2_bedroom_14-12.2.mkv` |
| 3 park, 20:51.4 | `output/t17/sa3/3_park_20-51.4_filled.mkv` | `output/t17/hunyuan/3_park_20-51.4_filled.mkv` | `output/t17/scenes/3_park_20-51.4.mkv`, and the library fill `output/t18b/3_typical_filled.mkv` |

Listen for: does it fit the place? Any music, melody or tones? Any voices?
Is it too loud or too quiet under the dialogue? Raw numbers:
`output/t17/sa3/results_cuda.json`, `output/t17/hunyuan/results_cuda_swap30.json`
and `results_cuda_swap36.json` (the Hunyuan files in the folder are from the
swap-36 run; swap 30 gave the same levels to 0.1 dB). Logs:
`output/t17/hunyuan_swap30.log`, `hunyuan_swap36.log`.

Setup notes from phase 2: the Hugging Face cache needed `refs/main` files
for SigLIP2 and CLAP so they load offline; `t17_sa3.py`'s memory probe
needed ctypes argument types. Nothing else changed. The project `.venv` was
not touched.

## Sources

- Stable Audio 3: [announcement](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models),
  [code and model table](https://github.com/Stability-AI/stable-audio-3),
  [model overview](https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/model-overview.md),
  [model card (gated)](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
  [mirror used](https://huggingface.co/cocktailpeanut/stable-audio-3-small-sfx),
  [Comfy-Org repack](https://huggingface.co/Comfy-Org/stable-audio-3).
- Stable Audio Open Small: [model card](https://huggingface.co/stabilityai/stable-audio-open-small).
- HunyuanVideo-Foley: [code, memory table](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley),
  [weights and licence](https://huggingface.co/tencent/HunyuanVideo-Foley),
  [fp8 / block-swap ComfyUI node](https://github.com/phazei/ComfyUI-HunyuanVideo-Foley),
  [fp8 weights](https://huggingface.co/phazei/HunyuanVideo-Foley),
  [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512),
  [CLAP](https://huggingface.co/laion/larger_clap_general).
- MMAudio: [code and licence](https://github.com/hkchengrex/MMAudio),
  [model list](https://github.com/hkchengrex/MMAudio/blob/main/docs/MODELS.md),
  [weights](https://huggingface.co/hkchengrex/MMAudio),
  [fp16 repacks](https://huggingface.co/Kijai/MMAudio_safetensors),
  [Apple DFN5B-CLIP licence](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384).
- ThinkSound / PrismAudio: [repo](https://github.com/QwenAudio/ThinkSound/tree/prismaudio),
  [model card](https://huggingface.co/FunAudioLLM/PrismAudio),
  [paper](https://arxiv.org/html/2511.18833).
- ControlFoley: [repo](https://github.com/xiaomi-research/controlfoley).
  Kling-Foley: [repo](https://github.com/klingfoley/Kling-Foley).
- Joint audio-video models: [MOVA](https://github.com/OpenMOSS/MOVA),
  [LTX-2](https://arxiv.org/html/2601.03233v1),
  [LTX Foley guide](https://docs.ltx.io/open-source-model/feature-guides/audio/video-to-audio-foley),
  [NAVA](https://github.com/ernie-research/NAVA), [Ovi](https://github.com/character-ai/Ovi),
  [FastVideo / FastH3](https://github.com/hao-ai-lab/fastvideo).
