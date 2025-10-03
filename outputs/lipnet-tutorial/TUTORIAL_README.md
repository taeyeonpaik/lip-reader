# Lip-reading video tutorial — TensorFlow edition

This project implements [Nicholas Renotte’s linked video](https://www.youtube.com/watch?v=uKyojQjbx4c): data loading, a TensorFlow pipeline, the CNN/Bi-LSTM model, CTC training and callbacks, checkpoint loading, and real-video predictions. It also includes a Streamlit companion demo.

**The real-video demo works now using the author’s published 96-epoch checkpoint.**
The separate local two-epoch training job has its own weights and status; see
[TRAINING_STATUS.md](TRAINING_STATUS.md). Loading the author’s checkpoint does not
mean that we trained those weights ourselves.

**Local training is complete:** two full epochs over 900 training and 100 validation
clips per epoch, taking 3.03 hours. Validation CTC loss decreased from 69.30 to
65.82. These early weights are undertrained: evaluation on all 100 validation
clips gave 99.5% word error and zero exact matches. See
`validation/local-two-epoch-evaluation.json`. The author's checkpoint remains the
default for useful demo predictions.

For uploads of new speakers, use the sibling **lipreading-studio** app at
[localhost:8502](http://127.0.0.1:8502). It adds automatic face/mouth alignment and
uses a separately credited pretrained Auto-AVSR model for varied English speech.
The tutorial's fixed mouth crop and single-speaker training are unsuitable for
general uploads.

## Run the demo

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

The package includes four source clips, reference transcripts, browser-playable
MP4s, animated mouth crops, and the converted inference weights. FFmpeg is needed
only to create new previews, not to use those already supplied.

To transcribe one sample without the app:

```bash
python -m tutorial predict \
  --weights models/tutorial.weights.h5 \
  --video samples/bras9a/bras9a.mpg \
  --alignment samples/bras9a/bras9a.align
```

Observed result: `bin red at s nine again`, matching the reference. Some other
clips have recognition errors; the app displays the reference so these are visible.
The model consumes only video pixels. The transcript is used after inference for
comparison.

## Follow the notebook

Open [LipReading.ipynb](LipReading.ipynb) in a notebook editor with the virtual
environment selected. Its eight code cells were executed successfully, including
real-video predictions. For a standalone notebook UI, install JupyterLab into the
environment with `python -m pip install jupyterlab`, then run `jupyter lab`.

The notebook's training cell is guarded by `RUN_TRAINING = False` to prevent
accidentally launching a second expensive job when reopening it. Set it to `True`
with a fresh output directory to train inside the notebook. Its saved outputs
show the status at execution time; `TRAINING_STATUS.md` is updated separately.

## Download the full tutorial data

```bash
python fetch_assets.py --output assets
```

This downloads the same Google Drive archives linked in the author's notebook:
approximately 423 MB of GRID speaker-1 data and 94 MB of raw checkpoints.
It extracts 1,000 videos and their alignments. The
[GRID corpus page](https://spandh.dcs.shef.ac.uk/gridcorpus/) describes the dataset
and its terms. Only the small demonstration subset is included in the project ZIP.

The author's raw checkpoint can also be loaded directly by using
`--weights assets/pretrained/checkpoint`.

## Train as shown in the video

The notebook specifies 100 epochs, but at **57:17–57:48** the presenter explicitly
runs only two for demonstration, then loads his 96-epoch checkpoint at about
**61:05**. This project supports either duration:

```bash
# Two complete epochs: 900 training and 100 validation clips, batch size two.
python -m tutorial train \
  --data assets/dataset/data --output runs/my-training --epochs 2

# Longer training from scratch.
python -m tutorial train \
  --data assets/dataset/data --output runs/long-training --epochs 100

# Resume an existing run to a larger total epoch count.
python -m tutorial train \
  --data assets/dataset/data --output runs/my-training --epochs 3 --resume
```

The command defaults to Adam at 1e-4, the tutorial's padded-label CTC objective,
and batch size two. It includes the learning-rate schedule, per-epoch predictions,
weight saving, full optimizer checkpoints, CSV history and progress reporting.
Checkpoint resumes restore weights, optimizer state, epoch and learning rate;
they do not promise bit-identical random dropout sequences.

Training this model is considerably slower on CPU than the GPU used in the video.
For a bounded test, add `--steps 1 --validation-steps 1`; this is explicitly a
**partial epoch**, not a substitute for the full dataset run. `--threads N` is a
global argument placed before `train`, `predict` or `evaluate`.

Run outputs:

- `run.json`: exact train/validation file lists and settings.
- `progress.json`: last recorded batch and loss, updated every ten batches.
- `history.csv`: completed epoch losses and learning rates.
- `last.weights.h5`: latest locally trained inference weights.
- `checkpoints/`: optimizer state for resume.
- `training-result.json`: written only after the requested fit returns.

## Evaluate and create previews

```bash
python -m tutorial evaluate \
  --data assets/dataset/data --weights models/tutorial.weights.h5 \
  --limit 10 --output validation/my-evaluation.json

python -m tutorial preview \
  --video samples/bras9a/bras9a.mpg --output previews

python -m pytest -q
```

The recorded ten-clip run had nine exact matches, 0.4% character error and 1.67%
word error. These are demonstration results on a small subset. The original
training split of the author's checkpoint is unknown, so these are **not
independent benchmark results**. The four examples featured in the app include
two correct predictions and two errors, recorded without filtering them out.

## Compatibility decisions

- Uses `tf_keras` with modern TensorFlow to read the tutorial's legacy checkpoints.
- Matches the tutorial model: convolution widths 128/256/75, two bidirectional
  128-unit LSTMs, and 41 output classes; 8,471,924 parameters.
- Preserves the author's BGR-as-RGB grayscale conversion and uint8 centering.
  They are unusual but changing them would alter what the checkpoint sees.
- Normalizes before padding shorter clips to 75 frames. Five of the supplied
  clips decode to 74 frames; the others decode to 75.
- Splits the file list deterministically before mapping it to videos. This keeps
  the same 900/100 sizes while avoiding accidental reshuffling of the split and
  slow decoding of skipped training clips to reach validation.
- Uses actual batch dimensions and resets the example callback's iterator so it
  cannot exhaust after repeated epochs.
- Greedy decoding matches the tutorial. Optional beam search uses TensorFlow's
  v2 decoder to preserve repeated letters separated by a blank.
- The exact tutorial CTC objective is the default. For new experiments,
  `--loss-mode actual-lengths` excludes label padding from the loss.
- Uses portable paths, explicit input checks, and separate output directories for
  downloaded weights and locally trained weights.

This is a tutorial reproduction, not a general-purpose lip reader for arbitrary
camera angles, languages or speakers. The fixed crop expects GRID-like source
geometry. It does not track moving mouths or transcribe arbitrary internet videos.

## Files and evidence

| File | Purpose |
| --- | --- |
| `LipReading.ipynb` | Executed walkthrough following the video sections |
| `tutorial/core.py` | Preprocessing, vocabulary, model, losses, decoders |
| `tutorial/workflow.py` | Data pipeline, training, evaluation, previews |
| `app.py` | Interactive local demonstration |
| `models/provenance.json` | Published-weight source and SHA-256 hashes |
| `validation/` | Tests, dataset audit, recorded predictions and screenshot |
| `COVERAGE.md` | Explicit implementation and execution coverage |
| `TRAINING_STATUS.md` | Full local run status |

The earlier sibling `lipnet` directory contains the separately implemented
PyTorch paper model. It is not checkpoint-compatible with this tutorial model.

## Sources

- [The user-provided video](https://www.youtube.com/watch?v=uKyojQjbx4c)
- [Author's notebook and companion app](https://github.com/nicknochnack/LipNet)
- [Underlying LipNet paper](https://arxiv.org/abs/1611.01599)

The source here was written for this project; the trained checkpoint and sample
data are attributed to their original sources. The published weights are included
for running the tutorial demonstration, with provenance kept alongside them.

## Current prototype: GRID practice (September 18)

The primary app has returned to this tutorial model. Launch `Start LipNet.command`
and open localhost:8502. It now explains the six-word sentence grammar, generates
random prompts, records one three-second webcam take, supports short uploads,
and shows raw plus grammar-constrained predictions. Prompt text is used only
for scoring after inference. `tutorial_demo.py` preserves the old TensorFlow demo.

The new app uses `requirements-grid.txt` and the existing Mac PyTorch environment;
TensorFlow remains necessary only for the original notebook/core workflows.
The verified PyTorch adapter keeps the tutorial architecture and author weights.
Automatic face alignment and grammar decoding are explicit additions.

A bounded adaptation run uses separate local training, validation and test
speakers. The tutorial checkpoint's full pretraining split is unknown, so these
results cannot certify completely unseen pretraining speakers. No extremely low
WER or physical-webcam accuracy is claimed before evaluation.
See [the current training recipe](training/GRID_RETURN.md).
