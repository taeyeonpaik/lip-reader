# LRS3-only full-model fine-tuning — revised validation protocol

## Current run: lrs3-only-v2

On September 20 the user explicitly chose to split the verified official-test
inventory into validation and final testing. The former validation clips came
from LRS3 trainval and could have been seen during author pretraining. Their
scores are historical diagnostics only and no longer select this run's model.

| Role | Clips | Source |
|---|---:|---|
| Training | 30,468 | Same verified LRS3 trainval rows and order as v1 |
| Validation | 296 | 82 recording groups from the cropped official-test inventory |
| Final test | 1,025 | Remaining 328 recording groups from that inventory |
| Unused | 1,488 | Former trainval validation partition |

No LRS2 samples enter this fine-tuning run. The original checkpoint retains its
mixed-corpus pretraining history. Official test is held out under the authors'
published protocol; we have not exhaustively audited every pretraining input.

## Deterministic split and limitations

`training/lrs3_protocol.py` joins all possible recording identities for ambiguous
clips into connected groups. It orders those groups by SHA256 using the fixed
salt `lrs3-official-holdout-v2-seed20260920` and assigns the first 20% (rounded
up: 82 of 410 groups) to validation. All 1,321 clips are retained. Possible
recording identities cannot cross partitions. Six ambiguous clips keep their
original ambiguity; no recording identity is guessed. The final partition has
328 groups covering 330 possible recording identities.

The split uses identifiers only, never predictions or errors. See
`runs/lrs3-only-v2/split-protocol.json` and `protocol-check.json` for independent
membership, order-invariance, ambiguity-bridge and checksum checks.

This is a custom validation/test split, not the complete official LRS3 benchmark.
The original pretrained model was previously evaluated on all 1,321 clips at
beam 5 (27.45% WER), so the final subset is not entirely untouched historically.
That earlier score does not select this split or decoder. From this change
onward, final-test predictions must not select weights, stopping or settings.

The original raw Oxford test release was unavailable; the user approved the
verified cropped mirror. Mouth crops do not reproduce the authors' original
full-face preprocessing exactly. These results are not claims of webcam accuracy.

## Sources

- Trainval: `TheNHz/ellipsis-lrs3-trainval-autoavsr`, revision
  `a9b5374e19082b3217171be8ece589f35c8a2a71`; independent author labels verified,
  with one unmatched source clip excluded before v1.
- Official-test inventory: `mattymchen/lrs3-test`, revision
  `9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a`; see
  `runs/lrs3-official-test-v1/dataset-verification.json`. All 1,321 transcripts
  and frame counts match independent author labels as a multiset. Raw face
  pixels are not verified.
- Source hashes depend on provenance, not current role: trainval hashes the
  inner `frames.npy` member; official-test-source rows hash the NPZ file,
  including rows now assigned to validation. Common pixel hashes check overlap.

## Fixed ordinary recipe (unchanged)

Original 250,383,410-parameter Auto-AVSR architecture; all parameters trainable.
Original checkpoint SHA256:
`fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`.

- AdamW lr `1e-5`, betas `(0.9, 0.98)`, weight decay `0.03`.
- Microbatch 1, accumulation 16, correct partial-group averaging, float32,
  gradient norm clipped to 1; one-epoch warmup and cosine decay over 20 epochs.
- Standard hybrid CTC/attention loss, author dropout and label smoothing.
- Author random crop 88×88 and AdaptiveTimeMask(10,25); evaluation center crop;
  normalization mean 0.421/std 0.165. Frozen BN running statistics, trainable affine.
- MPS frontend/encoder and CPU decoder/CTC with connected gradients.
- Maximum 20 total epochs, minimum 3, patience 4; no wall-clock cutoff.
- Original model is eligible as epoch-zero fallback. Lowest NEW validation WER
  selects the checkpoint; ties retain the earlier candidate.
- Fixed author beam 40, CTC weight 0.1, no external LM or length bonus, for
  validation and matched original/selected final testing. No decoder sweep.

## Preserving training progress

V1 was stopped safely in epoch two. Its atomic checkpoint contains epoch 2,
clip offset 23,920 and 3,400 completed optimizer updates. Original v1 checkpoints,
config, source and protocol document remain preserved. `migration-plan.json`
pins their hashes and the exact resume point.

V2 verifies the unchanged data, evaluates the original model on the new 296-clip
validation set, then evaluates the saved epoch-one candidate on that same set.
It resets selection/history validation metrics to those new results. It restores
v1's latest model AND optimizer, checks every model tensor matches, and writes
an atomic v2 checkpoint before continuing epoch two with the same learning-rate
schedule, training order and deterministic update seeds. The small unsaved part
of epoch two is replayed. `migration-check.json` records completed migration;
its absence during initial validation means migration is still in progress.

Epoch-one selection had been influenced by the old validation set; all completed
trained epochs (one) remain available and are re-scored under the new protocol.
Old trainval scores no longer participate in selection or early stopping.

## Progress and safe resume

- Current reports: `training/runs/lrs3-only-v2/`.
- Current log, relative to workspace: `work/lrs3-only/experiment-v2.log`.
- `progress.json`: live stage/count; `history.json`: completed epoch scores;
  `selection.json`: fixed selection written BEFORE any final-test evaluation.
- Large checkpoints: `/private/tmp/autoavsr-lrs3-only-v2`, outside iCloud.
  Preserve v1 recovery assets until migration and independent review are verified.
- Runtime: `/private/tmp/grid-clean-env/bin/python`.

The managed caffeinate job is already launched. Never start a duplicate. The
trainer holds an exclusive lock, verifies input and source/config hashes, and
writes atomic checkpoints every 100 updates and at epoch boundaries. Resume only
from consistent `last.pt` with unchanged current config/source. Do not edit the
running trainer. Verification uses eight workers and reads/decompresses each file
once. Checkpoint writes stop below 8 GiB free space.

For recovery only, from the project directory:

```bash
MPLCONFIGDIR=/private/tmp/lipnet-exact-mpl /private/tmp/grid-clean-env/bin/python -u -m training.lrs3_finetune
```

Keep the Mac plugged in and lid open. Caffeinate prevents idle sleep but does not
make closed-lid operation safe. Sustained v1 epoch one took about 6h16m including
its larger validation set. V2 uses fewer validation clips; actual timing will be
reported rather than promising a completion time. No extra training campaign,
new dataset, remote compute, spending or deployment is authorized.

## Completion review

1. Independently recompute new validation/final WER from predictions, verify
   membership, recording groups and any failures; inspect migration checks.
2. Verify original SHA remains unchanged, selected weights changed if trained,
   and fresh strict checkpoint loads reproduce predictions.
3. Quantify paired uncertainty using the predefined recording components;
   explain conservative grouping of ambiguous identities.
4. Report negative results honestly and label this custom cropped-test subset,
   not a complete official benchmark or general webcam performance.
5. Integrate only an eligible candidate after review and actual upload/recorded-
   webcam runtime checks. Preserve LipNet and author Auto-AVSR fallback.
6. Update review/start docs and compact source/report archive without full
   datasets or optimizer states. Record fine-tuning and evaluation completion.

No sub-15% WER is promised. No further tuning against the final partition.

## Paused September 20 — preserve for possible later resumption

The user requested a dataset-only pivot to LRS2. This LRS3 run is stopped at the
latest saved checkpoint: epoch 3, offset 1,440, update 3,900. All four existing
v1/v2 checkpoint files were moved to durable local storage outside iCloud with
original-path symlinks and verified hashes. See `runs/lrs3-only-v2/pivot-preservation.json`.
Best completed epoch: 2, validation WER 24.9777%; final testing has not run.
Do not resume this run automatically. The active LRS2 recipe is documented in
`LRS2_FINETUNING.md` and differs only in dataset handling and run paths.
