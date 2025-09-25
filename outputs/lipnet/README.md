# LipNet, implemented from scratch

**For the TensorFlow workflow from the linked video, see the separate
[video-tutorial project](../lipnet-tutorial/README.md).** It includes an executed
notebook, the author's trained checkpoint, real-video examples, and a local demo.
The package below is the PyTorch implementation of the underlying paper.

A runnable PyTorch implementation of the model behind the linked
[lip-reading tutorial](https://www.youtube.com/watch?v=uKyojQjbx4c), based on
[LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599)
by Assael, Shillingford, Whiteson and de Freitas.

**Status:** the first full run completed 200 epochs but missed its quality targets
(91.5% word error, no exact validation sentences). Its files remain in
`runs/full-training`. See [the diagnosis](validation/FIRST_RUN_REVIEW.md).

The corrected run is `runs/stabilized-training`; its current state and validation
results appear in the app. It adds feature LayerNorm, uses Xavier initialization
for recurrent input matrices, and reduces channel dropout to 0.25. The original
architecture remains the default; this revised recipe is a practical variant.
A four-clip learning check passed before launch; that is not a validation score.

## Complete Mac training run

`full_train.py` trains the 4,571,388-parameter paper architecture with mixed
precision on Apple MPS. The normalized variant has 4,574,844 parameters. CTC runs on the CPU with gradients transferred back to
the GPU. The tutorial's larger CNN/Bi-LSTM is implemented separately; TensorFlow
Metal cannot run its Conv3D layers on this Mac. The paper architecture measured
about 1.3 seconds per 12-clip training batch here.

The tutorial's 1,000-clip GRID s1 dataset is divided into 900 training and 100
validation clips using seed 42. RGB mouth images use crop `(80,190,140,46)`, resized
to 100×50, with channel statistics computed only from training data. Horizontal
flips augment training. Adam starts at 0.0003; a validation-loss plateau reduces
the rate. This is a practical training recipe, not the paper's full curriculum.

From this folder, on the current Mac:

```sh
../../work/venv/bin/python full_train.py \
  --data ../../work/tutorial-assets/dataset/data \
  --cache ../../work/paper-training-cache \
  --output runs/stabilized-training --epochs 200 --batch-size 12 --feature-norm --dropout 0.25
```

To recover an interrupted run, add `--resume` to the same command. Resume restores
model, optimizer, scheduler, mixed-precision scaler and random states. Do not run
two copies against the same output directory. The output lock prevents this.
To pause between batches, create `runs/stabilized-training/PAUSE`; delete it to resume.
The app provides buttons for these actions. The active launch prevents idle sleep
while it runs; closing the lid or stopping the process can still interrupt work.

Every epoch processes all 900 training and 100 validation clips. A checkpoint
qualifies when validation WER ≤10% and CER ≤4% for three consecutive epochs after
at least 40 epochs. Training ends at 200 epochs if this does not occur; the result
explicitly records whether the target was met. `best.pt` is selected by validation
WER, then CTC loss. `last.pt` supports resuming. These include optimizer state.
`best-validation.json` contains every validation prediction. Validation is used
for checkpoint selection, so these metrics are not an independent test score.

The app's **LipNet training → Try the locally trained checkpoint** reads held-out
GRID samples with these actual local weights. Custom uploads and webcam use the
separate, pretrained Auto-AVSR model, which supports broader English speech.

## Quick start

From this directory, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m lipnet smoke --output runs/smoke
```

The smoke command uses a small model with the same layer structure. To exercise
the full model in a short optimisation run:

```bash
python -m lipnet smoke --full-model --steps 5 --output runs/full-smoke
```

The `lipnet` console command is equivalent to `python -m lipnet` after installation.
Run `python -m lipnet --help` to list commands. CPU and CUDA are supported; `auto`
selects CUDA if available and CPU otherwise. This original CLI uses CPU on Apple
Silicon; the new `full_train.py` runner above explicitly supports MPS training.

## What is implemented

- RGB video decoding, configurable mouth crop, 25 fps checks, normalization.
- GRID alignment parsing and deterministic speaker-aware train/evaluation splits.
- Three 3D convolution blocks, spatial pooling and channel dropout.
- Two bidirectional GRU layers with an explicitly implemented recurrence.
- Character classifier, CTC objective, greedy and log-space prefix beam decoding.
- Horizontal flip and frame deletion/duplication augmentation.
- Training, validation, corpus CER/WER, gradient clipping and resumable checkpoints.
- Tests for CTC alignment, exact decoder probabilities, padding, gradients and video I/O.

The network and decoding logic are original code. PyTorch provides tensor
operations, autodiff, convolution kernels, the CTC loss kernel and Adam; OpenCV
provides video decoding. This is not a framework-free implementation.

## Train on GRID

### 1. Obtain and organize data

Get video archives and word alignments from the
[official GRID corpus page](https://spandh.dcs.shef.ac.uk/gridcorpus/).
Follow its dataset terms. Audio archives are unnecessary. Extract to directories
that contain speaker components named `s1`, `s2`, etc. For example:

```text
data/
  videos/
    s1/bbaf2n.mpg
    s3/bbaf2n.mpg
  alignments/
    s1/bbaf2n.align
    s3/bbaf2n.align
```

Additional directories below each speaker directory are allowed. Each video is
matched to its alignment using both speaker and filename stem. Duplicate matches,
missing alignments and unsupported transcript characters cause explicit errors.

### 2. Prepare mouth crops

The network expects mouth regions, not whole faces. Choose either:

- **Pre-cropped videos:** supply 25 fps RGB mouth clips. They are resized to 100×50.
- **Fixed camera and speaker position:** pass `--crop X Y W H` in source-frame
  pixels to `check`, `stats` and `train`. Inspect your videos to choose the box.

Omitting `--crop` uses the entire supplied image as the mouth region. A fixed box
is inappropriate when the face moves outside it. For varying head poses or
multi-speaker data, create tracked, aligned mouth crops externally first. This
package does not ship the paper's DLib/iBug landmark tracker or Kalman filter.
Use identical preprocessing throughout a run; it is saved inside checkpoints.

Videos with the wrong frame rate, an invalid crop, no frames, or over 300 frames
are rejected. Resample or segment them beforehand. Clips are never silently
truncated to fit their transcripts.

### 3. Build manifests

```bash
python -m lipnet prepare \
  --videos data/videos --alignments data/alignments \
  --output data/split --split unseen
```

`unseen` holds out speakers s1, s2, s20 and s22. `overlapped` holds out 255 random
clips per speaker using `--seed 42`; use `--heldout N` for small development subsets.
Neither mode claims to recover the authors' exact random file lists.

`eval.jsonl` is the benchmark holdout. **Create a separate validation manifest from
the training pool for checkpoint selection; keep the benchmark holdout untouched.**
The following recipe produces `fit.jsonl` and `val.jsonl` without touching it:

```bash
python - <<'PY'
import random
from lipnet.data import read_manifest, write_manifest
rows = read_manifest('data/split/train.jsonl')
if len(rows) < 2:
    raise ValueError('Need at least two training clips')
random.Random(42).shuffle(rows)
n = max(1, len(rows) // 10)
write_manifest(rows[n:], 'data/split/fit.jsonl')
write_manifest(rows[:n], 'data/split/val.jsonl')
PY
```

Manifests are one JSON object per line. Custom datasets can use the same format:

```json
{"video": "../mouths/clip.mp4", "text": "set blue by a four please", "speaker": "s3"}
```

Relative paths resolve against the manifest's directory. The preparation command
writes absolute paths; regenerate manifests if you move the dataset. Text supports
letters a–z and spaces; case/whitespace are normalized and digits must be spelled out.

### 4. Validate data and compute training statistics

For already cropped videos:

```bash
python -m lipnet check --manifest data/split/fit.jsonl
python -m lipnet check --manifest data/split/val.jsonl
python -m lipnet stats --manifest data/split/fit.jsonl --output data/stats.json
```

For full-frame videos, append your chosen `--crop X Y W H` to all three commands
and the training command below. `stats` uses pixel-weighted RGB moments from the
training subset only. The default normalization values come from the paper, but
new crops and cameras should use their own training statistics.

### 5. Train and resume

```bash
python -m lipnet train \
  --train data/split/fit.jsonl --val data/split/val.jsonl \
  --stats data/stats.json --output runs/grid \
  --epochs 100 --batch-size 4 --device auto
```

Batch size 4 is a practical default, not the paper's batch size 50. Increase it if
memory permits. Training from scratch on the full corpus is substantial work and
should use a CUDA GPU. The Python GRU recurrence favors readability and equation
fidelity over fused-kernel speed. Epoch count 100 is an example, not a convergence
guarantee. `--threads 4` is a global option and must precede the subcommand.

Training writes `last.pt`, `best.pt` (lowest validation CTC loss), and
`history.jsonl`. Checkpoints include the model, vocabulary, preprocessing,
optimizer and random states. To continue to a total of 150 epochs:

```bash
python -m lipnet train \
  --train data/split/fit.jsonl --val data/split/val.jsonl \
  --output runs/grid --resume runs/grid/last.pt --epochs 150
```

On resume, the saved preprocessing and optimizer learning rate take precedence;
omit `--crop` and `--stats`. Keep batch size, augmentation, datasets and worker
count consistent for reproducibility. Exact equality across hardware or PyTorch
versions is not promised. Seeded CPU resume is tested with zero data workers.

### 6. Evaluate and transcribe

```bash
python -m lipnet evaluate \
  --manifest data/split/eval.jsonl --checkpoint runs/grid/best.pt \
  --beam-width 200 --output runs/grid/evaluation.json

python -m lipnet predict \
  --video data/videos/s1/bbaf2n.mpg --checkpoint runs/grid/best.pt
```

Use input video geometry matching training. Both commands restore the checkpoint's
preprocessing automatically. `--beam-width 1` selects fast greedy decoding; larger
values select prefix search. Metrics are corpus-weighted fractions: CER counts
character edits (including internal spaces), WER counts word edits, and sentence
accuracy counts exact matches. WER can exceed 1 because insertions count as errors.

## Architecture and scope

Dimensions below omit the batch axis. All convolution/pooling layers preserve time.

| Stage | Output |
| --- | --- |
| RGB mouth input | 3 × T × 50 × 100 |
| Conv 3×5×5, spatial stride 2; pool 1×2×2 | 32 × T × 12 × 25 |
| Conv 3×5×5, stride 1; pool 1×2×2 | 64 × T × 6 × 12 |
| Conv 3×3×3, stride 1; pool 1×2×2 | 96 × T × 3 × 6 |
| Flatten each timestep | T × 1728 |
| Bi-GRU, 256 per direction, twice | T × 512 |
| Linear | T × 28 |

Convolutions use ReLU and 0.5 channel dropout after pooling. Initialization uses
He weights and orthogonal recurrent matrices. The recurrent equations follow
section 3.2, including applying the reset gate before the recurrent candidate
matrix multiplication. CTC blank has ID 0; letters have IDs 1–26 and space ID 27.

**Architecture-table ambiguity:** appendix A.2 lists spatial stride 2 for later
convolutions while its adjacent tensor sizes imply stride 1. This implementation
uses stride 1 for convolutions two and three, preserving the listed 1728-feature
GRU input. This is an explicit interpretation of an inconsistent table.

**Differences from the full published experiment:** external mouth tracking is
required; individual-word curriculum augmentation and the character 5-gram
language model are not implemented. Prefix search sums visual-model CTC
probabilities without language-model scoring. Frame perturbation interprets the
paper's 0.05 probability as 0.025 deletion and 0.025 duplication. Training defaults
to smaller batches, adds gradient clipping, and selects checkpoints by validation
loss. These choices and missing experimental components mean this package must
not be presented as a reproduction of the published benchmark scores.

## Code map

| File | Responsibility |
| --- | --- |
| `lipnet/model.py` | CNN, custom GRU, CTC validation/loss |
| `lipnet/data.py` | Video decoding, transcripts, splits, augmentation, statistics |
| `lipnet/text.py` | Vocabulary, CTC decoders, edit metrics |
| `lipnet/engine.py` | Training, evaluation, checkpointing, smoke test |
| `lipnet/cli.py` | User commands |
| `tests/` | Mathematical, model and pipeline tests |
| `VALIDATION.md` | Checks actually run in the development environment |

## References

- [Paper, including architecture appendix](https://arxiv.org/html/1611.01599v2)
- [User-provided video tutorial](https://www.youtube.com/watch?v=uKyojQjbx4c)
- [Official GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)

This package uses PyTorch rather than the linked tutorial's TensorFlow stack.
It does not load the tutorial's TensorFlow checkpoints.
