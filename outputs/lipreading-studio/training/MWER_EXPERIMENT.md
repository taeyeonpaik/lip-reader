# Candidate-ranking diagnostic and MWER pilot — experiment 5

## Scope and status

Authorized September 18: one final bounded attempt while the user sleeps. Run
only on this Mac. Do not deploy or purchase compute. If there is no convincing
improvement, report that and stop; do not start another experiment.

## Final MWER pilot completed — September 18

The final bounded attempt finished at **07:59 EDT**, after approximately
**76 minutes** of diagnosis, training and evaluation. Both conventional training
and MWER completed **two full epochs and 250 updates each** on the same 1,000
training clips, with identical initialization and data order. Each arm updated
26,670,450 decoder/CTC parameters; the visual encoder stayed frozen.

### Validation results (400 clips, 200 per dataset)

| Model | LRS2 WER | LRS3 WER | Mean of dataset WERs | Total word errors |
| --- | ---: | ---: | ---: | ---: |
| Pretrained, beam 20 | 25.83% | 10.99% | 18.41% | 638 |
| Conventional, epoch 1 | 25.83% | 10.76% | 18.29% | 633 |
| Conventional, epoch 2 | 25.51% | 10.58% | 18.04% | 624 |
| MWER, epoch 1 | 25.83% | 10.85% | 18.34% | 635 |
| MWER, epoch 2 | 25.57% | 10.67% | 18.12% | 627 |

**Outcome:** MWER corrected 11 word errors overall, reducing mean WER by **0.289
percentage points**. The matched conventional-training control corrected 14,
reducing mean WER by **0.367 points**. MWER did not outperform the control.
Neither met the predefined 0.50-point improvement requirement. No candidate was
selected, and the conditional additional audit/test evaluations were not run.
These results do not establish an improved test model.

The diagnostic found a better candidate among the top ten token sequences for
160 of 400 clips. Choosing candidates with the reference answers gave **11.32%
mean WER**, versus 18.41% for actual top-ranked predictions. This is an oracle
score: it uses the answers and is NOT deployable model accuracy. The diagnostic
supported trying ranking-focused training, but the pilot captured little of that
potential. The result does not prove that all possible MWER recipes would fail.

Conventional training loss decreased from 3.509 to 3.206. MWER's centered
objective decreased from −0.249 to −0.328; negative values are expected because
candidate error counts are centered. Loss values across these different
objectives are not directly comparable.

The app retains the original pretrained weights and fast beam-5 decoding.
Its previously measured **16.19% LRS2 test WER** is unchanged. All pilot comparisons
use beam 20 and a different validation subset, so their numbers must not be
compared directly with that test score or the larger experiment-4 validation.

**This is the final training attempt under the user's instruction.** No further
training campaign or deployment is planned. The app remains usable locally,
and all candidate checkpoints, split provenance, predictions and experiment
code are retained for review. Dataset media is excluded from the app archive.


## Question and diagnostic

Does the model generate better transcripts but rank them below the top result?
On 400 validation clips (200 per dataset), generate up to ten distinct token
sequences with beam 20 and no external LM. Compare the actual top-ranked
transcript with the candidate having fewest reference-word errors. This latter
"oracle" score uses the answer and is only an estimate of candidate-list
headroom; it is NOT a usable model or an achievable accuracy claim.

Proceed to the pilot only if the mean of per-dataset oracle WERs is at least two
percentage points lower than the actual top-ranked mean WER. Otherwise finish
with a diagnostic report; do not force an unrelated visual-encoder campaign.

## Fixed pilot

- Start each arm from the same original Auto-AVSR checkpoint.
- Training: 1,000 clips, 500 LRS2 and 500 verified LRS3.
- Diagnostic/selection validation: 400 clips, 200 per dataset.
- Subsequent audit: 800 different validation clips, 400 per dataset.
- The 1,243 LRS2 test clips are used only after selection and a consistent audit
  improvement. Existing validation/test have been examined in previous work;
  these are not newly unseen benchmarks. No official LRS3 test is included.
- Deterministic selection: sort each dataset/split by SHA256 of
  `mwer-v5-45:` plus namespaced clip ID. First 500 training clips; first 200
  validation clips for diagnosis/selection, next 400 for audit. IDs and media
  hashes are published in the run's split manifest, without raw media.
- Freeze the entire visual encoder and reuse the full feature cache. Train the
  last two attention-decoder blocks, final decoder normalization/output, and CTC
  projection. Dropout is disabled in BOTH arms, with gradients enabled.
- **Two epochs per arm**, effective batch 8, AdamW learning rate 1e-5, weight
  decay .01, gradient norm clip 1. Same training order and seed per epoch.
- **Control arm:** ordinary .9 attention / .1 CTC loss used by this model.
- **MWER arm:** expected word-error count over the N-best list, plus .1 times
  the ordinary loss. Recompute differentiable sentence scores for every update.
  Sentence log score = .9 attention log probability (including EOS) + .1 CTC
  log probability. Softmax normalizes over candidates; subtracting the unweighted
  mean error count does not change the gradient.
- N-best lists are generated from epoch-start weights and held fixed within each
  epoch, then regenerated for epoch 2. This bounded approximation avoids running
  beam search at every optimizer update. References are NOT injected into lists.
- No learning-rate, interpolation-weight, beam, or epoch-budget sweep follows.

This adapts [Prabhavalkar et al., Minimum Word Error Rate Training](https://arxiv.org/abs/1712.01818)
to Auto-AVSR's visual-only, joint attention/CTC scoring. It is not a reproduction
of that paper's audio model, data, hyperparameters or published gains. The
regularizing conventional loss also includes this model's label smoothing/CTC.

## Selection before audit

A checkpoint must reduce mean per-dataset validation WER by at least **0.5
percentage points**, have pooled CER no worse than baseline, and worsen neither
dataset's WER by more than **0.2 points**. This last tolerance is predefined for
this pilot so a few errors on one subset do not automatically veto an otherwise
useful gain. Report all per-dataset changes, not just the mean.

Choose the qualifying arm/epoch with lowest mean WER before reading audit/test
results. Report the control-versus-MWER comparison explicitly: a control gain
alone is not evidence that MWER helped. After selection, evaluate the fixed
candidate and baseline on the additional audit clips. Only a consistent mean
WER gain with no pooled CER regression proceeds to test. Do not change settings
or pick another epoch using audit or test results.

Run `validation/check_mwer_completion.py` from workspace root with the project Python after completion. Before any app activation, independently recompute metrics, estimate uncertainty
using paired recording-session bootstrap, verify checkpoint tensors and reload
predictions, and measure the selected configuration's upload/webcam throughput.
Preserve the original model and fast fallback. A noisy/inconclusive result is
reported as such; do not promise sub-10% WER or launch another run.

## Engineering verification

Unit checks cover MWER gradient direction, constant-baseline gradient invariance,
equal-error zero gradients, invalid inputs, and selection limits. The real-data
smoke check covers both RGB LRS2 and grayscale LRS3: batched versus individual
candidate scores, agreement with inference beam scores (including EOS and CTC),
finite gradients and actual parameter changes. The real-data checks passed; evidence is in
`validation/mwer-feasibility.json`. All 26 regression tests passed.

## Run and recovery

From workspace root:

```sh
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.mwer \
  --manifest work/corpora/verified-manifest.json \
  --cache work/overnight/features \
  --run outputs/lipreading-studio/training/runs/mwer-v5
```

Use a managed foreground process and redirect output to
`work/mwer/experiment.log`. Do not duplicate a running process. An exclusive run
lock prevents two copies. Completed diagnostic/beam rows and completed epochs
are reusable. An interrupted epoch replays from its last completed checkpoint;
partial validation from that epoch is discarded. Preserve epoch-start N-best
lists. Configuration and base hash must match on resume. Do not call a run
complete merely because a tool session disappeared; inspect progress and logs.

Saved candidate bundles use format 3 with base hash, exact decoder prefixes,
state, arm, epoch and validation metrics. The app loader currently supports
format 1 only; add strict format-3 support ONLY if a candidate passes review.
Never load a decoder-only bundle through the visual/deeper format-2 loader.

## Completion and retention

The pilot is complete. Both arms and all four epoch checkpoints are retained.
Neither qualified for the conditional audit/test step. Original weights and the
fast app decoder remain active; no format-3 adapter was activated. Independent
metric, checkpoint and uncertainty evidence is published in
`validation/mwer-checkpoint-and-metrics.json`. The final review and ZIP include
this experiment. No further training is authorized without a new user request.

## MWER verification and uncertainty

All five evaluation reports and the candidate-oracle score were independently
recomputed. Reloading all four epoch checkpoints reproduced eight recorded
predictions (one per dataset per checkpoint). Every targeted module changed,
and the original checkpoint hash remained intact.

A paired bootstrap over recording sessions (2,000 repetitions, stratified by
dataset) estimated these 95% intervals for **mean WER reduction**, in percentage
points:

- Conventional versus pretrained: **+0.053 to +0.743** (estimate +0.367).
- MWER versus pretrained: **+0.017 to +0.600** (estimate +0.289).
- MWER versus conventional: **−0.405 to +0.203** (estimate −0.078).

The first two intervals support small gains on this validation sample. They do
not remove bias from choosing the better epoch on the same validation data, or
establish unseen-speaker/test improvement. The comparison between methods gives
no evidence that MWER is superior. The predefined minimum gain was not met, so
we preserve the original app model and finish the attempt as requested.

All 26 regression tests passed before the run; the active recognition pipeline
is unchanged. Previously verified upload/WebRTC throughput therefore remains the
relevant runtime evidence. No candidate was activated or deployed.

