# LipNet — original implementations

**Fine-tuning and independent review are complete.** Uploads and webcam practice default to the reviewed epoch-4 checkpoint, with an original pretrained selector and integrity-checked fallback. Validation WER improved **20.33% → 18.44%**; final-test WER improved **18.00% → 17.25%** on the fixed local sample. The test improvement is modest and statistically inconclusive. See [REVIEW.md](REVIEW.md) and [FINETUNING.md](FINETUNING.md).

This local project runs the **original notebook model** and the **rizkiarm paper repository model** as separate implementations. Their architectures, preprocessing and decoders differ. The notebook uses published author weights; the paper model was fine-tuned locally from the author’s unseen-speaker weights. The previous custom PyTorch alignment/adaptation pipeline is historical and is not loaded here.

## Start

Double-click **Start LipNet.command**, then open [the local app](http://127.0.0.1:8502). On this Mac the isolated runtime is `/private/tmp/lipnet-exact-tf`. It uses the CPU and no remote service. Nothing is deployed.

If the temporary runtime is removed, install Python 3.13 and FFmpeg (including `ffprobe`), then run **Setup Local.command**. The bundled dlib wheel was built for this Mac's Python 3.13/macOS 26 environment. Other systems require a compatible dlib build. Full tested package versions are in `requirements-tested.txt`.

## Use

**Auto-AVSR is also available:** select **Auto-AVSR · unrestricted English** in the sidebar for natural-English uploads and eight-second webcam segments. Both LipNet options remain available. See [AUTOAVSR.md](AUTOAVSR.md) for usage and the optional local dependency.

- **Original examples:** run Nick's original notebook on bundled GRID clips, or the paper model on its supplied examples. References are shown only after prediction.
- **Upload:** one forward-facing person, one GRID sentence, 25 fps, 1–5 seconds and at most 640 pixels on the longest side. The original GRID recordings are 360 × 288. The app applies the paper repository's own dlib mouth crop and decoder.
- **Webcam practice:** generate a sentence, enable the camera controls, start a three-second take, then wait for transcription. The practice sentence never enters recognition. No microphone is used and webcam recordings are not saved. This is capture-then-transcribe, not continuous streaming recognition.
- **Verified results:** local prediction reports and their limits. Results distinguish the published baselines from the locally fine-tuned checkpoint. Neither model was trained from scratch here.

The GRID format is **command + color + preposition + letter + number + ending**. Example: `bin red at s nine again`. All choices appear in the app. The letter W is excluded, as in GRID.

## Source fidelity

| | Nick's notebook | Rizkiarm repository |
|---|---|---|
| Actual source executed | Notebook model/loader/loss cells | `lipnet/model2.py`, `videos.py`, original loss and spelling code |
| CNN widths | 128, 256, 75 | 32, 64, 96 with batch normalization |
| Recurrent network | Two bidirectional LSTMs, 128 per direction | Two bidirectional GRUs, 256 per direction |
| Input | Fixed crop, 75 × 46 × 140 grayscale | Original dlib crop, T × 100 × 50 RGB / 255 |
| Output classes | 41 | 28 |
| Published checkpoint | Tutorial 96-epoch weights | Repository `unseen-weights178.h5` |
| Decoder | Original greedy CTC | Original beam 200 + dictionary spelling correction |

Both upstream sources are pinned and included, with SHA-256 hashes in `reports/source-manifest.json`. Compatibility changes required by current Python/TensorFlow are documented in [COMPATIBILITY.md](COMPATIBILITY.md). The paper model has completed six fine-tuning epochs. Its decoder retains beam 200 and original spelling scores with deterministic alphabetical tie ordering; see the compatibility notes.

See [REVIEW.md](REVIEW.md) for measured results, runtime and limitations, and [PROVENANCE.md](PROVENANCE.md) for attribution.

## Training code

`train.py` provides explicit training entry points using each source's model, preprocessing, loss and learning rate. Original repository training scripts and the notebook are also included unchanged. The modern paper data adapter replaces the old multiprocessing generator; this is not a bit-for-bit recreation of its complete training run.

Run from this project folder, with the tested environment active:

```sh
python train.py tutorial --data /path/to/data --epochs 100 --initialization random --output /path/to/new-run
python train.py paper --data /path/to/mouth-dataset --epochs 178 --initialization random --output /path/to/new-run
```

Tutorial layout: `data/s1/*.mpg` and `data/alignments/s1/*.align`. Its original batch-two, padded-target loss and take450/skip450 split semantics are preserved, including the original shuffle behavior. This split is not an unseen-speaker evaluation. Paper layout: `train/speaker/clip/*.png`, `val/speaker/clip/*.png`, `align/clip.align`, with original 100 × 50 mouth frames and batch 50. Full original training is expensive on a CPU; these commands do not run automatically.

Both training paths passed a disposable gradient/optimizer-step check. The subsequent bounded fine-tuning run trained on 1,192 usable s3/s4 clips for six epochs (61.6 minutes of optimization), selecting epoch 4 using s1 validation before evaluating s2 test. Original author weights remain unchanged. This is completed fine-tuning, not a rerun of the full paper training procedure.

## Recheck

```sh
python check_compatibility.py
python check_runtime.py
python audit_reports.py
python check_spelling_determinism.py
python check_model_selection.py
```

`verify.py` additionally evaluates the local GRID files. From the parent workspace, set `PYTHONPATH=outputs/lipnet-exact` and run `python outputs/lipnet-exact/verify.py tutorial --count 100` or `paper --count 100`. The workspace's verified `work/grid-return/data/s1` and `s2` files are needed; the full dataset is excluded from the ZIP. The deterministic selection, every reference/prediction, failures and raw/processed metrics are saved. No model or decoder is selected by these results.
