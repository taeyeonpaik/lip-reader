---
license: cc-by-4.0
task_categories:
  - automatic-speech-recognition
language:
  - en
tags:
  - lip-reading
  - visual-speech-recognition
  - lrs3
  - webdataset
pretty_name: LRS3 trainval — auto_avsr mouth crops as WebDataset shards
---

# LRS3 trainval — auto_avsr mouth crops, bridged to WebDataset

The LRS3 **trainval** split (30 h) as `[T,96,96] uint8` grayscale mouth-ROI shards with character
transcripts, produced with **auto_avsr's own crop geometry** so results stay comparable to that
project's published LRS3 numbers.

Derived from [`TheNHz/ellipsis-lrs3-raw`](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw)
(itself a verified mirror of LRS3-TED, whose official distribution was discontinued).

## Attribution

LRS3-TED is by **Triantafyllos Afouras, Joon Son Chung and Andrew Zisserman** (VGG, University of
Oxford), distributed under CC BY 4.0:

> T. Afouras, J. S. Chung, A. Zisserman. *LRS3-TED: a large-scale dataset for visual speech
> recognition.* arXiv:1809.00496, 2018.

The crop geometry and the published landmarks come from **auto_avsr** (Pingchuan Ma et al., Imperial
College London, Apache 2.0):

> P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, M. Pantic. *Auto-AVSR: Audio-visual
> speech recognition with automatic labels.* ICASSP 2023.

## Contents

| | |
|---|---|
| Clips | **31,957** (of 31,982 — 25 dropped, see below) |
| Duration | **30.07 h** @ 25 fps (2,706,466 frames) |
| Clip length | median 2.96 s, min 0.48 s, max 6.20 s |
| Character set | 38 symbols: `A–Z`, `0–9`, space, apostrophe |
| Shards | 32 WebDataset tars, 25.0 GB |

Each sample is three consecutive tar members: `<key>.roi.npy` (`[T,96,96]` uint8 grayscale),
`<key>.txt` (the transcript), `<key>.json` (provenance: `prompt_id`, `session_key`, `fps`,
`n_frames`, `lang`). `<key>` is a zero-padded index; the real clip identity lives in the JSON.

## How the crops were produced

1. Source video: LRS3 trainval face tracks (224×224, 25 fps).
2. Landmarks: mpc001's published `LRS3_landmarks` — no face detection is re-run, so the geometry is
   auto_avsr's rather than a reimplementation.
3. Crop: auto_avsr's `VideoProcess` (`detectors/retinaface/video_process.py`) — 68-point similarity
   warp to their mean face, 96×96 mouth patch, `window_margin=12` landmark smoothing.
4. Grayscale: applied **after** the warp with BT.601 weights (0.299/0.587/0.114), matching
   auto_avsr's published order — their preprocessing warps in RGB and applies
   `torchvision.transforms.Grayscale()` at training time. Storing uint8 quantises once (≤1/255).

To feed a model trained on auto_avsr's pipeline, apply their serve transform:
`/255 → CenterCrop(88) → Normalize(0.421, 0.165)`.

**25 clips are absent** because auto_avsr's `VideoProcess` returns no patch for them (fewer frames
than the smoothing window, or degenerate landmark geometry) — the same clips its own preprocessing
loop skips. Their IDs are in `bridge-report.json`.

## Verification

- Every clip loads to `[T,1,88,88]` float32 through the serve transform.
- The auto_avsr `vsr_trlrs3_base.pth` visual encoder yields finite, **non-collapsed** embeddings:
  across 100 sampled clips the least-varied has mean pairwise cosine distance **0.745** between
  frames (a collapsed encoder would sit near 0) — the crops are in-distribution for it.
- Shard bytes are reproducible: identical `version_hash` across runs with different worker counts.

`dataset-manifest.jsonl` carries the content hash
`sha256:3534d9b915cff59d1ca38166f19da5d2e7eabb02eea4c0a429d91b9680fd06a3`.

## Caveats

- **This is trainval only** — the 407 h pretrain split is not included. Compare against published
  numbers for the equivalent low-resource (30 h) setting, not against full-corpus results.
- Transcripts are LRS3's own, parsed exactly as auto_avsr parses them (braces stripped, words kept).
- No audio: these shards are for visual-only recognition.
