# LRS2 CUDA migration — September 20, 2026

Local LRS2 training was stopped for severe memory pressure. Its temporary
checkpoint disappeared after the Mac restarted. Restart from the original
Auto-AVSR weights, not from the preserved LRS3 candidates. Local reports remain
historical; no claim of resuming the lost optimizer state.

Remote root: /teamspace/studios/this_studio/lipreader
Project: outputs/lipreading-studio
Trainer: training/lrs2_cuda.py
Run: training/runs/lrs2-cuda-v1
Persistent checkpoints: lipreader/checkpoints/lrs2-cuda-v1

Same architecture, 45,814/1,082/1,243 LRS2 partitions, float32, AdamW recipe,
augmentation, hybrid loss, accumulation16, frozen BN statistics, and beam40.
CUDA training replaces MPS/CPU placement; beam search remains CPU. TF32 disabled.
Cloud PyTorch runtime differs from Mac and must be reported with results.
All input hashes and decoded crop properties checked in the cloud before training.
Validation selects checkpoint; test is evaluated only after selection is fixed.
Baseline is recomputed, never copied from a different execution environment.

CPU setup commands from the project directory:

    python -u -m training.lrs2_cuda --verify-only
    python -u -m training.lrs2_cuda --check-only

After the user selects/approves a GPU and applicable cost, run discarded CUDA
feasibility updates, then start the real run (fresh pretrained weights):

    python -u -m training.lrs2_cuda --smoke
    python -u -m training.lrs2_cuda

Do not automatically provision a GPU, start paid compute, deploy, or restart the
Mac trainer. A CUDA readiness check is still required before actual training.
SIGINT/SIGTERM during optimization request an atomic save at the next complete
optimizer update. Hard termination may replay work since the last periodic save.
Resume is strict about unchanged source, config, tokenizer, vendor and manifests.
No full media or optimizer checkpoint belongs in a public project archive.

## Verified setup result

All 48,139 videos passed source SHA, decoded-frame and split checks. Original
weights passed SHA verification and strict model loading. CPU loss and sample
beam40 prediction passed; dependency check reports no broken requirements.
GPU feasibility and training remain pending; no GPU has been provisioned.
See local `work/lightning/reports/` for copied verification records.

## T4 compatibility test passed

The full 250,383,410-parameter float32 model passed finite-gradient updates,
representative weight changes, a strict saved-checkpoint prediction reload and
an optimizer-resume update on Tesla T4. All diagnostic updates were discarded;
validation and test clips were not used. Original pretrained weights unchanged.

A 16-clip training-length-stratified sample gave a warmed median 0.1589 s/clip,
projecting 2.02 hours per training epoch before validation, checkpoints, or
full-dataset I/O effects. This is a short benchmark, not measured epoch runtime.
Peak reserved GPU memory in the tested batches was 5.04 GiB (allocated 4.67 GiB).
Full training has not started. Only the bounded $0.55 test was authorized; get
a total GPU spending limit before launching a longer run.

## Batched trainer implementation

The user authorized four-clip microbatches (effective batch 16), parallel CPU
preprocessing/prefetch, and length bucketing. Implemented separately as
`training.lrs2_batched`, retaining the original trainer. Eight correctness tests
and actual pretrained inference comparisons passed. No GPU speed result yet.
Studio is stopped; uploading the prepared source bundle remains pending.
See [BATCHING.md](BATCHING.md) for evidence and controlled handoff steps.

## A100 short benchmark

FP32: 0.12257 seconds/clip, projected 1.56 training hours/epoch.
BF16 (FP32 CTC and loss): 0.14509 seconds/clip, projected 1.85 hours/epoch.
Both passed finite-gradient checks; all updates discarded. These short-sample
projections exclude validation and full-dataset I/O overhead. At the earlier
2.19/hour list price FP32 projects about 3.42 credits/training epoch; the actual
interruptible rate was not verified. Cost break-even versus the measured T4
at 0.55/hour is approximately 0.713 credits/hour for this A100 speed.

Automatic CPU switch was rejected because the LambdaLabs cluster lacks cpu-4.
The stop fallback succeeded: SSH gateway confirms the Studio is not running.
Persistent files remain. Do not restart a GPU without further authorization.
The complete report is work/lightning/a100-test-result.json in the Studio;
local a100-review.json records values observed from its printed test log.
