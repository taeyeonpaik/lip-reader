# GRID LipNet final review — 19 September 2026

## Outcome

**The bounded adaptation finished, but the low-WER/new-speaker goal was not achieved.**
The prototype has the requested grammar guide, random generator, upload inference,
and three-second webcam takes. Its current automatically aligned recognition is
not accurate enough to describe as a reliable webcam lip reader.

The author’s original 96-epoch tutorial checkpoint remains active. Three local
adaptation epochs ran; none passed the fixed validation rule. No extra training
campaign was started. Nothing was deployed. Earlier experiments remain preserved.

## Measured results

All numbers below use automatic face alignment, the fixed 75-frame input and
beam-64 grammar decoding where stated. These are not the original fixed-crop
paper benchmarks. WER is word error rate; lower is better.

| Model | Split | Clips | Raw WER | Grammar WER | Exact sentences |
|---|---|---:|---:|---:|---:|
| Author checkpoint | Validation, s2 | 150 | 87.11% | 80.00% | 0 |
| Epoch 1 | Validation, s2 | 150 | 88.00% | 80.56% | 0 |
| Epoch 2 | Validation, s2 | 150 | 90.44% | 81.00% | 0 |
| Epoch 3 | Validation, s2 | 150 | 91.00% | 81.22% | 0 |
| Retained author checkpoint | Final test, s4 | 200 | 92.92% | 81.25% | 0 |

Final-test grammar errors: 975 / 1,200 reference words; raw errors: 1,115 / 1,200.
Rejected adapted models were not selected or tuned using final-test results.
The total run took 50.6 minutes, including preprocessing and evaluation.
Training loss fell from 166.93 to 78.51 while validation WER failed to improve.

A 10,000-resample clip bootstrap gives a final-test grammar-WER interval of
79.17–83.33%. The paired epoch-3-minus-author validation difference is +1.22
percentage points, with a clip-bootstrap interval of −0.44 to +3.00 points.
These intervals are **conditional on the single speaker in each split**; they
cannot measure uncertainty across new speakers or independent recording sessions.
There are too few held-out speakers for a meaningful speaker bootstrap.

## What was verified independently

- A separate full-matrix edit-distance implementation recomputed all five reports.
- All 950 manifest transcripts match independently decoded GRID clip filenames.
- Train s1/s3, validation s2, and test s4 have disjoint speaker IDs and video hashes.
- All archive checksums were checked before training. The ambiguous s5 archive
  was excluded in full because its names matched s6 alignment files, not s5.
- Original initialization SHA-256 remains unchanged. Twenty state tensors changed
  in the last checkpoint; all tensors are finite. Strict checkpoint loading worked.
- Three validation predictions from the author model and three from the reloaded
  epoch-3 model exactly matched their saved raw and constrained predictions.
- The app’s selection file still selects the author checkpoint. No failed adapted
  weights were activated. The practice prompt is used only after recognition.

Evidence and reproducible checker: `validation/grid/final-review.json` and
`training/review_grid.py`. The checker performs no training and never changes
model selection.

## Runtime and prototype checks

Five focused tests passed. Browser checks covered sentence generation, original
example inference, webcam controls and training/result display. Two recorded
three-second examples exercised the actual upload conversion and one-take worker.
Upload processing, including conversion, took 0.78–1.82 seconds; recorded webcam
worker inference took 0.41–0.42 seconds. These are small local smoke measurements,
not a broad latency benchmark or a physical-camera test. Recognition still made
errors despite these processing times. No physical camera was activated/tested.

The original fixed-crop `bras9a` demonstration gives “bin red at s nine again.”
Automatic alignment of the same example changes its prediction. This indicates
that the added alignment path can hurt predictions; this experiment does not
isolate how much of the poor held-out result comes from alignment versus speaker
generalization. No preprocessing changes were selected using final-test results.

## Limits and provenance

This is the Nicholas Renotte tutorial’s CNN/BiLSTM architecture, executed by a
weight-compatible PyTorch adapter on the Mac. The secondary rizkiarm paper
implementation uses GRUs and is a different architecture. Automatic alignment,
GRID grammar decoding and actual-length CTC training are documented extensions.

The author checkpoint’s complete training history is unknown. Local adaptation
speakers are disjoint, but global pretraining speaker independence is not certified.
Neither tutorial demonstrations nor these local results establish webcam accuracy.
The app should not be presented as a high-accuracy résumé result on new speakers.

The original TensorFlow project and older experiments remain in the workspace.
The compact GRID ZIP contains current runtime/training/review code, original
inference weights, reports and a small existing demo set. It excludes full GRID
media and optimizer checkpoints. `outputs/GRID_PACKAGE_VERIFICATION.json` records
its final SHA-256 and verifies every archived member. No further automatic training
is scheduled after final review; changes to the experiment require a new decision.
