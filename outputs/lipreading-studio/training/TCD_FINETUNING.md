# TCD-TIMIT full-model fine-tuning

Status: 2026-09-23 — user approved proceeding with all 59 volunteers. The three
professional lipspeaker archives failed repeated HTTP 503 requests and are excluded.
CPU transcript inventory and crop preparation are in progress; no TCD training or
model evaluation has started.

## Acquisition

Official description: https://sigmedia.tv/datasets/tcd_timit/
Official portal: https://sigmedia.tcd.ie/
Recordings: https://sigmedia.tcd.ie/tcd_timit_db/volunteers
The recordings URL redirects to a login form. Register/sign in through the portal;
do not put passwords in chat or source code. Respect the non-commercial terms.
Do not substitute a third-party crop mirror without reviewing its source/labels.

## Fixed approach

- Preserve LRS2, LRS3, LipNet and original author weights/checkpoints.
- Start from original Auto-AVSR author weights, not the LRS2 fine-tuned candidate.
- Use frontal recordings of the 59 volunteers; exclude Lipspkr1–3 per user approval.
  Preserve original population labels and namespace speaker IDs. Do not report
  results for professional lipspeakers.
- Freeze speaker-separated train/validation/test lists before any model predictions.
  Inspect published speaker-independent splits first. If a custom split is needed,
  document its actual speaker IDs/counts; do not call it an official benchmark.
- Use authoritative word transcripts; never synthesize references from model output.
- Preprocess full videos into tracked/aligned 96x96 RGB mouth crops at 25 fps,
  preserving utterance boundaries; retain source and crop hashes. Confirm this on
  actual samples before bulk processing. Decode failures must be audited, not silently dropped.
- Reuse the current trainer unchanged: all parameters trainable, AdamW 1e-5,
  betas (.9,.98), weight decay .03; microbatch 4, accumulation 4; FP32, frozen BN
  statistics, original hybrid loss and augmentation, beam 40; maximum 20 epochs,
  minimum 3, patience 4. Best validation WER selects weights, including baseline.
- Test the original and selected models only after selection; calculate WER, CER,
  exact sentence accuracy and paired speaker-bootstrap uncertainty.
- No guarantee of improvement or sub-15% WER. Scores across datasets are not direct comparisons.

## Implementation

`training.tcd_finetune` subclasses the existing batched trainer without modifying
LRS2 code or run directories. It requires `work/corpora/tcd-timit/manifest.json`
and `provenance.json`. Each row contains id, split, verified source speaker ID (e.g. 01M), population
`volunteer` or `lipspeaker`, view `straightcam`, path, text, transcript_source, declared_frames,
source_sha256 (raw media), and sha256 (prepared crop). Provenance must include
dataset `TCD-TIMIT`, source, transcript_source, preprocessing, split_policy and
review_complete=true, only after the data audit actually passes.

Run/checkpoints: `training/runs/tcd-timit-v1`, `checkpoints/tcd-timit-v1`.
The adapter rejects speaker leakage and duplicate media crossing partitions.
Input verification uses the existing full-file hash and decoded pixel checks.
No launcher has been deployed; no GPU allocated. After acquisition and validation:

```bash
python -m training.tcd_finetune --verify-only
python -m training.tcd_finetune
```

## Remaining before training

1. Completed: obtain 59 official volunteer archives and bundled word references.
2. Actual archive layout inspected; custom speaker split pinned; frame-rate audit pending.
3. Implement/verify crop conversion and transcript alignment on real samples.
4. Pin manifests/provenance, audit leakage and preprocessing, verify model inputs.
5. Verify cloud storage and available credits before GPU launch; benchmark memory
   on the longest training clips without silently excluding long utterances.
6. Launch a single resumable process, preserving existing run assets.

After training: independently recompute metrics, check checkpoint reload and
unchanged original weights, measure upload/recorded-webcam runtime, update app
only after review, and publish compact reports/source without raw recordings.

## Download selection (updated from portal screenshots)

Use straightcam.zip once per speaker for both populations. Volunteer straightcam/
folders duplicate the archive; do not fetch both. Exclude 30degcam files. Include
README.txt and both volunteer_labelfiles.mlf and lipspeaker_labelfiles.mlf.
The shown archives are 4.7 GB for one volunteer and 27.3 GB for one lipspeaker;
these are not total dataset sizes. Inventory and storage budgeting are required
before bulk download. Download/extract/crop per speaker to avoid retaining all
full-resolution archives and extracted copies simultaneously. 59 volunteer archives have now been downloaded and verified; professional archives
remain incomplete and are excluded.

## Volunteer-only preparation

`work/lightning/prepare_tcd.py` reads the bundled word-level TXT references and
uses the existing app mouth-tracking pipeline. A custom split of 41 training,
9 validation and 9 test speakers is pinned with seed 91 before model inference.
Published split SCP resources have not been obtained from the authenticated
Additional Resources folder; this is not claimed as an official benchmark split.
Shared sentence prompts across speaker partitions are counted and reported.
Archives remain intact; only one raw MP4 is extracted temporarily per conversion.
Every crop has raw-media and output SHA256, source transcript attribution, frame
count and tracking coverage. Failures are logged and block final review; the
provenance review flag remains false until the actual data audit passes.

Word references contain one `start end word` line per token. The parser strips
timestamps on every line, preserves apostrophes, and rejects malformed lines.

## Final preparation and evaluation accounting

Preparation produced 5,781 valid crops from 5,782 source clips. One test clip,
`volunteer/56M/sx67`, failed tracking twice with unchanged settings. It is retained
in `evaluation-failures.json` with its source word reference. Both baseline and
selected test reports include its empty prediction (all reference words count
as deletions); the final test denominator is 882, not 881. No model prediction
was used to decide this handling.

Counts: 4,018 training crops / 882 validation crops / 881 usable test crops plus
one test preprocessing failure. Custom 41/9/9 speaker split. The source inventory
reports 588 distinct sentence texts shared across partitions; this evaluates
held-out speakers, not wholly unseen sentence prompts.

`work/lightning/audit_tcd.py` independently checks all crop hashes, decodes every
frame, validates shape/frame rate/count, checks normalized word references and
speaker/pixel separation. Only a successful full audit marks provenance reviewed.
The TCD evaluator has a resume test confirming failed references are counted once,
with all their words as deletions, and never passed into model inference.

GPU training remains pending current hardware availability and the user's
current spending cap; original model initialization and recipe remain unchanged.
