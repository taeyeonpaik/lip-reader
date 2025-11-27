# Local pretrained English adaptation

> **Completed September 17:** Deeper fine-tuning finished 10 epochs and 15,000 updates. Validation WER improved from 21.40% to 21.06%, while test WER worsened from 16.19% to 16.25%. The app retains the pretrained model. [Full experiment results](DEEPER_EXPERIMENT.md). The earlier output-layer experiment is also documented below.

This is a real, bounded fine-tuning experiment for **Auto-AVSR**, the model used
by Lip Reading Studio. It is not LipNet training and does not reproduce the
original paper's training procedure.

## Completed result

Three full epochs and 1,500 optimizer updates completed in about 39.5 minutes.
Validation WER did not improve (21.40% baseline; 21.40%, 21.54%, 21.48% by epoch),
so early stopping retained the pretrained model. The unchanged model achieved
16.19% WER and 10.24% CER on 1,243 prepared-mouth test clips, with 728 exact
transcripts and no failures. Real fine-tuning completed; no accuracy improvement
was established.

## Experiment

- Base: authors' `vsr_trlrs2lrs3vox2avsp_base.pth`, 250,383,410 parameters.
- Adapted: CTC and attention output projections, 7,765,362 parameters (3.1%).
- Frozen: visual frontend, Conformer encoder, decoder body; dropout remains off.
- Data: real LRS2 transcripts and 96×96 RGB mouth-crop videos from the public
  MahmoodAnaam split distributions on Hugging Face.
- Split sizes: 4,000 training / 500 validation / all 1,243 clean test clips.
- Training subset: downloaded shards 0 and 11 of 22; deterministic hashed ID
  ordering fixes the subset before any predictions. Validation is similarly
  selected from the separate validation split.
- Optimizer: AdamW, learning rate 3e-6, weight decay 0.01, effective batch size
  eight via gradient accumulation, gradient norm clipped to one.
- Objective: author's 0.1 CTC + 0.9 label-smoothed attention loss; true transcripts.
- Budget: up to eight full epochs; stop after three epochs without a better
  qualifying validation result. Random seed 42.
- Selection: validation word error must improve by at least 0.5 percentage points
  and character error must not worsen. Otherwise retain the original model.
- Test: evaluated only after selection is fixed; never used to tune parameters.

Features are cached in float32 because the visual encoder is frozen. The encoder
runs on Apple MPS; the two trainable projections and decoder run on CPU. The
original one-gigabyte checkpoint is never overwritten. Adapters are approximately
31 MB and explicitly tied to the original checkpoint hash.

## Status and evidence

`runs/lrs2-heads-v1/progress.json` reports the active stage and last update.
`history.json` records completed epochs. A complete experiment writes `result.json`
with baseline and selected-model test metrics. `last.pt` retains the optimizer and
output layers for recovery. Separate JSON and JSONL prediction files provide the
reference, hypothesis, errors and timing for every evaluated sample.

The app's **Model & training** tab displays progress and final results. The earlier
three-update feasibility check used arbitrary labels; that diagnostic is retained
in `../validation/pretrained-adaptation-feasibility.json` and is not an accuracy result.

## Data provenance and limits

[Data provenance](data-provenance.json) records repository revisions, file hashes,
selection rules and split checks. There are no duplicate sample IDs, identical
video bytes across splits, or overlapping recording IDs in the selected data.
Speaker identity beyond recording IDs is not known. We retain the original split
assignments rather than claiming a newly verified speaker-independent benchmark.

The base model was already trained on LRS2. A local training or validation gain
therefore is not proof of generalization to new speakers. The final test comes
from the distributor's clean official-test split; the source paper's exact
pretraining membership has not been independently reconstructed.

These files are already aligned mouth crops. This benchmark evaluates recognition
and decoding, **not face detection**. Tracking failures and webcam transport are
covered separately by application tests. The app uses beam width five; the
reference research decoder uses a wider beam. Dataset videos are kept in the
workspace cache and excluded from the application archive.

## Reproduce on this Mac

From the project workspace root, with `work/venv` already installed:

```sh
# Fetch the exact public split files, then create and verify the local manifest.
work/venv/bin/python outputs/lipreading-studio/training/download.py --destination work/overnight/data
PYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.data \
  --downloads work/overnight/data/downloads.json --destination work/overnight/dataset

# Complete the experiment, or resume the same configuration from its checkpoint.
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.experiment \
  --manifest work/overnight/dataset/manifest.json \
  --cache work/overnight/features \
  --run outputs/lipreading-studio/training/runs/lrs2-heads-v1
```

On another machine, install the app requirements plus `training/requirements.txt`
and use that Python executable. A run directory accepts only the exact original
configuration. For a new experiment choose a new run directory. Keep the test set
untouched until the design and validation selection are final.

Recovery resumes from the last complete epoch; an interrupted partial epoch is
repeated with the same deterministic order. Cached feature extraction and
prediction JSONL logs avoid repeating completed work. Only one process can hold
a run's filesystem lock.

## Interpreting the numbers

Word error rate is `(substitutions + deletions + insertions) / reference words`.
It can exceed 100% and is not the same as sentence accuracy. Character error
includes normalized spaces and apostrophes. Text is uppercased; punctuation other
than apostrophes is replaced with spaces. Failed examples count as empty
predictions, so their reference words contribute deletion errors.

Sources: [Auto-AVSR](https://github.com/mpc001/auto_avsr),
[LRS2 overview](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/),
[training](https://huggingface.co/datasets/MahmoodAnaam/LRS2-Train),
[validation](https://huggingface.co/datasets/MahmoodAnaam/LRS2-Validation),
[test](https://huggingface.co/datasets/MahmoodAnaam/LRS2-Test).
