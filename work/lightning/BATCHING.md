# Batched LRS2 training — September 20, 2026

## Status

Uploaded and verified in the CPU Studio. At the user’s explicit request, the
production run was launched on the selected A100 80 GB, skipping the separate
speed benchmark. All 48,139 source hashes passed. Baseline validation precedes
epoch 1 automatically. See `batched-training-launch.json`; no epoch speed result
or completed training update is claimed by this launch record.

A pre-gradient metadata fix converts the PyTorch version to a builtin string so
weights-only checkpoint reload works. The new regression test passed locally;
metadata reload also passed directly on the GPU host. The original eight CPU
Linux tests passed before this fix. The numerical training recipe is unchanged.
The launcher stops the Studio when the training process exits.

## Changes

- Four clips per GPU forward/backward pass; four passes per optimizer update.
  Effective batch remains 16. The final six clips use correctly weighted 4+2
  microbatches, with no dropped or duplicated examples.
- Two CPU workers decode, crop and normalize upcoming clips. Each worker
  prefetches two batches; pinned memory and nonblocking transfers feed CUDA.
  A 60-second worker timeout makes loader failures explicit.
- Randomized pools of 256 clips are sorted by length before forming four-clip
  batches. Full batches are then shuffled. Every training clip appears once.
- True frame/label lengths mask encoder and decoder attention and CTC. Temporary
  hooks also zero padded activations before Conformer depthwise convolutions;
  attention masks alone would allow convolutional padding contamination.

The original 250M architecture, weights, train/validation/test splits, full-model
training, AdamW recipe, schedule, frozen BN statistics, hybrid loss, augmentation
distribution, beam40 validation, early stopping and final-test policy are retained.
Precision stays FP32 with TF32 off: the earlier A100 BF16 experiment was slower.
No vendor/model parameter definitions changed. One-clip evaluation is unchanged.

Batch order and random-number allocation intentionally change. This does not
promise identical future weights or WER. Per-clip augmentation is now keyed to
seed/epoch/ID, so worker scheduling and prefetch do not change it on resume.

## Verification

- Eight CPU tests passed in 24.44 seconds using PyTorch 2.8.0. They cover padding
  invariance, original versus batched losses and all parameter gradients, partial
  updates, exact-once data coverage, frozen BN, author augmentation semantics,
  two-worker versus single-worker preprocessing, and exact checkpoint resume
  through the new training loop with dropout enabled.
- Actual original checkpoint: inference-only comparison on four LRS2 **training**
  clips (13/18/22/30 frames). Maximum encoder difference 0.000014663; mean hybrid
  loss difference 0.000005119. No production gradients were run on the Mac.
  Original model SHA unchanged. See `batched-model-check.json`.
- On all 45,814 training entries, first-epoch bucketing adds 1.47% padded frames,
  versus 59.66% for random four-clip batches. It processes 36.44% fewer frames
  than those random batches. This is **not** a measured runtime speedup versus
  the original single-clip trainer, which does not pad. See `batching-data-check.json`.
- CUDA memory, kernel behavior, and actual throughput remain to be tested.
  Under-one-hour epochs are not established. Validation/checkpoint time is extra.

The first sandboxed worker test stalled on macOS multiprocessing; it was stopped
and the complete suite passed with normal shared-memory permissions. The isolated
CPU test environment is `/private/tmp/lipreader-batch-tests`, disposable after
verification. It contains no model checkpoints or dataset copies.

## Cloud handoff

`batched-update.zip` contains only the new source, tests, compact evidence and
instructions. Extract into `/teamspace/studios/this_studio/lipreader`, preserving
paths. It excludes data, model weights, optimizer states and environments.
`batched-update-verification.json` lists hashes. Existing cloud dependencies
already support the trainer; pytest is needed only to run the test suite.

The standalone trainer is:

```bash
cd /teamspace/studios/this_studio/lipreader/outputs/lipreading-studio
python -m training.lrs2_batched --verify-only
```

This CPU-only step repeats the original data checks in the new run directory;
perform it before allocating a GPU. It does not train.

After a GPU and test spending cap are approved, the prepared benchmark wrapper is:

```bash
cd /teamspace/studios/this_studio/lipreader
bash work/lightning/run_batched_test.sh
```

The wrapper caps its Python process at 120 seconds (plus a five-second kill
grace), including load/probe overhead. It probes a four-clip longest-video batch,
then performs three discarded 16-clip updates with CPU prefetch and FP32. It
checks finite gradients, weight changes and the original SHA, and writes
`batched-test-result.json`. No validation/test examples are used. It attempts to
return to CPU, otherwise stops the Studio. Provisioning/shutdown/billing time
are outside the Python timer; it is not a guaranteed dollar cap.

After benchmark review and a training spending limit, launch:

```bash
cd /teamspace/studios/this_studio/lipreader/outputs/lipreading-studio
nohup python -u -m training.lrs2_batched > ../../work/lightning/batched-training.log 2>&1 < /dev/null &
```

Reports: `outputs/lipreading-studio/training/runs/lrs2-batched-v1/`.
Atomic model/optimizer/progress checkpoints: `checkpoints/lrs2-batched-v1/`
in persistent Studio storage. SIGTERM requests a checkpoint after the current
complete optimizer update. Resume requires identical configuration/source and
verified data. Interrupted epoch-end validation reuses the saved epoch weights.

No production cloud training checkpoint existed when this version was prepared;
earlier T4/A100 diagnostic updates were discarded. This version starts from the
unchanged original author checkpoint. The old `training.lrs2_cuda` and its reports
remain available. Preserved LRS3 checkpoints and LipNet are untouched.

## CUDA validation activated

At the user's request to accelerate validation, beam40 decoding now stays on the
A100, using one CPU orchestration thread. The old Mac-derived evaluation path
moved the decoder/CTC to CPU. No clips, beam settings, weights, loss or training
batch settings changed. Twelve completed validation examples (stratified by
length) reproduced their earlier predictions. Mean GPU decode time was 1.0015s
versus 3.0874s for the earlier complete prediction path; the probe excludes GPU
encoding time, so this is not a measured full-pass speedup. Single-thread CPU
was slower and was rejected. No test examples were used.

426 completed baseline predictions were preserved. The process was restarted
before any gradient updates, with an execution-only source/config migration.
Remaining baseline predictions and subsequent validation/test passes use CUDA.
Sampled agreement is not a proof of identical CPU/GPU floating-point results on
all utterances. See validation-cuda-check.json and cuda-validation-migration.json.

## Worker termination recovery

The trainer and a DataLoader worker received SIGTERM during epoch 1. The source
of the signal is unconfirmed. The last valid checkpoint held 40,000 clips / 2,500
updates. Every model tensor, optimizer tensor and progress value was verified
unchanged during a metadata-only source fingerprint migration; the original
checkpoint remains as `checkpoints/lrs2-batched-v1/before-worker-signal-fix.pt`.
Workers now allow 60 seconds after SIGTERM for the parent to finish its optimizer
update and save. Normal queue-based worker shutdown still applies; the alarm
bounds orphan lifetime. The supervisor now uses absolute Python/CLI paths.
No hyperparameters, model weights, splits or decoding settings were changed.

Three targeted local tests passed (signal grace, exact training-loop resume,
safe checkpoint reload), and the signal-grace probe passed on the Linux host.
The Studio reported auto_sleep=true and timeout=600. Lightning documents that
active scripts should continue when the browser closes; inactivity causation
was not established and account/subscription settings were not changed.
