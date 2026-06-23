# Active run — A100 LRS2 batched training

The user authorized immediate production training on the selected A100 80 GB.
All source hashes passed. The run evaluates the original validation baseline,
then trains automatically. No separate GPU speed benchmark was run.

Progress: `outputs/lipreading-studio/training/runs/lrs2-batched-v1/progress.json`.
Log: `work/lightning/batched-training.log`.
Checkpoints: `checkpoints/lrs2-batched-v1/last.pt` and `best.pth`.
Validation decoding now uses the A100 at unchanged beam40. The first 426
baseline predictions were preserved when switching from CPU decoding.
The supervisor stops the Studio on exit. Do not launch a duplicate.
See [BATCHING.md](BATCHING.md) and `batched-training-launch.json`.

## Earlier setup notes (historical)

# LRS2 training on Lightning

## Current status — batched implementation ready locally

Studio is stopped following the A100 test. The new four-clip trainer, parallel
loader and length bucketing have passed local correctness tests. See
[BATCHING.md](BATCHING.md) for the verified results and upload/benchmark steps.
Cloud upload and CUDA timing remain pending. The older setup notes below are
historical; do not launch the one-clip trainer or paid compute automatically.

Setup is running on CPU. Open `lipreader/work/lightning/setup.log` or use:

```bash
tail -f /teamspace/studios/this_studio/lipreader/work/lightning/data-restore.log
```

The data is reconstructed from exactly the same pinned public shards; every
video hash, ID, transcript, length and split is compared to the Mac manifest.
Original pretrained weights are hash verified. All optimization restarts from
those weights because the Mac's temporary LRS2 checkpoint was lost on reboot.

Persistent results and checkpoints:

- `lipreader/outputs/lipreading-studio/training/runs/lrs2-cuda-v1/`
- `lipreader/checkpoints/lrs2-cuda-v1/`

Do not change the architecture, splits, hyperparameters, or decoder based on test
results. Do not touch preserved LRS3 or LipNet. Do not deploy the app.

After CPU readiness passes and the user approves the GPU/cost, select the GPU in
the Studio toolbar. Run the training-only feasibility check first:

```bash
cd /teamspace/studios/this_studio/lipreader/outputs/lipreading-studio
python -u -m training.lrs2_cuda --smoke
```

This discards its two diagnostic updates. On success, start actual training:

```bash
nohup python -u -m training.lrs2_cuda > ../../work/lightning/training.log 2>&1 < /dev/null &
tail -f ../../work/lightning/training.log
```

Check `progress.json` and per-epoch validation reports in the run directory.
The best validation checkpoint is selected before the final test. No claim of
sub-15% WER, general webcam accuracy, or exact original preprocessing is made.

A single process lock prevents duplicate work. SIGTERM during optimization saves
at the next completed accumulation group. Wait for the paused progress state
before stopping the Studio. Sudden shutdowns can lose updates since the last
periodic atomic checkpoint; they do not delete persistent checkpoint files.
Cloud hardware execution has not been validated until the CUDA smoke check passes.

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
