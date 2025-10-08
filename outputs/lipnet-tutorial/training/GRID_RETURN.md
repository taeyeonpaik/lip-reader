# Return to tutorial LipNet — GRID practice

## Authorized scope

The user chose the original Nicholas Renotte tutorial and GRID's limited sentence
structure, prioritizing **new speakers**. Restore the local prototype, add sentence
instructions and a random generator, and aim for low WER. Do not deploy or use
remote/paid compute. Earlier unrestricted experiments are preserved as history.

## What is implemented

- Same tutorial 3D CNN widths 128/256/75, two Bi-LSTMs (128 per direction), and
  41 character outputs, with the author's supplied 96-epoch weights.
- Existing TensorFlow notebook/core remain intact. PyTorch adapter preserves
  architecture and published weights; prior CPU/MPS probability differences were
  below 1.3e-6. New NumPy input parity on `bras9a` is below 1e-6. This is the tutorial
  model, not rizkiarm's distinct Bi-GRU paper implementation.
- CTC prefix beam search (fixed width 64) accepts the GRID grammar. It has no
  transcript/prompt input. Raw greedy predictions are shown alongside it.
- Six slots: command, color, preposition, letter (excluding W), number, ending.
  The generator samples the full 64,000-sentence grammar. Targets enter only
  post-recognition practice scoring.
- Webcam captures one 3-second take; uploads accept one 1–5-second sentence.
  RGB face detection, eye/mouth alignment to a tutorial reference template,
  fixed mouth crop and temporal resampling produce 75 input frames. This is an
  explicit extension to the original fixed-coordinate GRID loader. Audio is not
  captured or used. Camera access is a user-controlled action.
- The original sample-only TensorFlow app is preserved as `tutorial_demo.py`.

## Data and evaluation boundaries

Official GRID archive: https://zenodo.org/records/3625687 . Four speaker ZIPs and
alignments were downloaded into `work/grid-return`, verified against published
MD5 hashes, and safely extracted. Speaker 1 remains from the original tutorial
archive. Provenance and split identities will be saved without dataset videos.

Fixed deterministic recipe, seed 91:
- Train: 300 clips each from s1 and s3 (600 total).
- Validation: 150 clips from s2.
- Final test: 200 clips from s4, used only after validation selects the model.

**The tutorial checkpoint's complete original training history is not published.**
These are disjoint speakers for local adaptation, not a certified unseen-speaker
benchmark relative to every stage of author pretraining. Do not claim otherwise.
The earlier 10-clip 1.67% WER was an author-checkpoint demonstration, not a benchmark.

## Bounded adaptation

`training/grid_experiment.py` updates all 8,471,924 effective tutorial parameters.
Adam lr 1e-5, eps 1e-7; microbatch 2, accumulation 4; gradient clip 5; float16 CNN
and float32 LSTMs/CTC on MPS. It uses actual transcript lengths, deliberately
correcting the tutorial's padded-target CTC loss. This is adaptation, not an exact
reproduction of the original training arithmetic or weights trained from scratch.

Maximum six epochs or two hours of training, checked between full epochs; minimum
three epochs before patience two. Preprocessing and final evaluation are additional.
A candidate must improve constrained validation WER by at least 0.5 percentage
points without worsening raw WER. Best valid constrained WER wins, raw WER breaks
ties. Otherwise keep the author weights. Final test never changes selection.
No new sweep or additional campaign is automatically authorized.

Run from the workspace root:

```sh
PYTHONPATH=outputs/lipnet-tutorial MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i /private/tmp/grid-clean-env/bin/python -u -m training.grid_experiment \
  >> work/grid-return/experiment.log 2>&1
```

An exclusive lock prevents duplicate training. Check process state before any
restart. Resume unchanged using verified preprocessed caches and `last.pt`
(model, optimizer, scaler, data-order seed and completed epoch). A 4 GiB disk floor
protects cache creation. No test-driven selection or preprocessing adjustment.

## Final status — 19 September 2026

Completed three epochs and stopped on validation patience. No candidate passed
validation, so the original author checkpoint remains active. Independent review
recomputed all five reports, checked 950 transcript identities and split hashes,
verified 20 changed state tensors and six reloaded predictions, and measured
upload/recorded-webcam runtime. No physical-camera test was performed.

Final retained-model test WER: **81.25% grammar / 92.92% raw**, zero exact sentences
on 200 clips from s4. Validation grammar WER was 80.00% for the author and 81.22%
at epoch 3. The low-WER goal was not achieved. See [REVIEW.md](../REVIEW.md) and
`validation/grid/final-review.json` for uncertainty and limitations.

Training is no longer running. Do not restart the completed experiment or create
another training campaign automatically. The app stays local on 127.0.0.1:8502.
The final package is built with `work/grid-return/package.py`; final verification
is `outputs/GRID_PACKAGE_VERIFICATION.json`. Monitoring ends after packaging.

## Recovery history

Runtime note: the isolated environment lives in temporary local storage. If it
is missing after reboot, recreate it with Python 3.13 and the pinned versions in
`requirements-grid.txt`; do not use a remote runtime. No cloud sync services or
system-wide settings were changed.

Preflight correction: the published s5 ZIP has 1,000 filename matches with s6 alignments and zero with s5. It is excluded in full; identity is unresolved. Before any preprocessing/training/evaluation, training was revised to 300 clips each from s1 and s3. Original preflight config is preserved. Validation/test and all optimization settings are unchanged. Candidate publication does not activate it until independent metric/runtime review.

Runtime recovery: dataless cloud placeholders caused old sample/dataset reads to stall or return EOF. Recovered s1 and demonstration MPG/transcripts from CRC-verified work/tutorial-assets/data.zip into work/grid-return/data. See work/grid-return/s1-recovery.json. Managed training session is now 3862; verify live PID before recovery. App browser generator and original-example inference work.

Current preflight checks: five tests pass (1.11s); browser generator changes the prompt, example inference and live progress render correctly. Author runtime sample: fixed crop correct; automatic upload and recorded-frame webcam predictions have errors. Recorded webcam worker processed 3s in 0.42s; first upload including conversion 3.55s (single sample, not a throughput benchmark). Physical camera not tested. Evidence: validation/grid. Package builder: work/grid-return/package.py; rebuild after final review.

The same AMP/CTC/Adam recipe passed a finite-gradient and optimizer-step smoke check on two cached training inputs; no saved weights were modified. See validation/grid/gradient-smoke.json. The initial prototype ZIP was marked training_complete=false. The final reviewed ZIP is rebuilt with completed results and is marked training_complete=true. The completion heartbeat is removed after package verification.
