# LRS2-only full-model Auto-AVSR fine-tuning

## Cloud migration supersedes the local recovery instructions below

On September 20 the Mac job was stopped because of severe memory pressure.
After reboot, its `/private/tmp` checkpoint was no longer present. Local logs and
baseline remain historical; the local optimizer state cannot be resumed. No
completed local epoch is being presented as a cloud result.

The authorized replacement is `training/lrs2_cuda.py`, with the same recipe and
LRS2 membership, restarted from original author weights. New run: `lrs2-cuda-v1`.
Cloud project: `/teamspace/studios/this_studio/lipreader/outputs/lipreading-studio`.
Checkpoints: `/teamspace/studios/this_studio/lipreader/checkpoints/lrs2-cuda-v1`.
These are persistent Studio files. Do not restart the old Mac job automatically.
See `work/lightning/README.md` and `recipe-migration-check.json` in the workspace.
Setup and data verification occur on CPU; GPU activation awaits user approval.
The preserved LRS3 checkpoints and LipNet remain untouched.

## Authorization and current status

On September 20 the user requested a pivot from LRS3 to LRS2, changing only the
dataset while preserving LRS3 checkpoints and work. The LRS3 process is stopped.
The LRS2 job has launched and verifies the complete local input inventory before
baseline validation and training. This is full-model fine-tuning, not the earlier
7.8M-parameter output-head or 55M-parameter partial-model experiments.

## Dataset

| Role | Local clips | Oxford published count |
|---|---:|---:|
| Train | 45,814 | 45,839 |
| Validation | 1,082 | 1,082 |
| Test | 1,243 | 1,243 |

Oxford's page: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html
The original sets are divided by broadcast date. The raw download and official
filelists returned HTTP 401 and require a BBC-approved data-sharing agreement and
password. No access gate was bypassed. Reuse the existing public processed mouth
crops rather than reacquiring 50 GB of raw media or changing preprocessing.

Sources already downloaded and hashed:

- MahmoodAnaam/LRS2-Train, revision `26dea093e50d1ee89fe717648b50922cc4b4011b`.
- MahmoodAnaam/LRS2-Validation, revision `5f987b9729b85720a81b2b521491cc204c19ed6c`.
- MahmoodAnaam/LRS2-Test, revision `e14bf18868ca78bd97b0b9ce6615b262a32dfdf1`.

`work/corpora/lrs2/manifest.json` supplies all rows in their original distributor
partitions, without sampling or moving validation/test rows into training. All
48,139 local files are present. There is no clip-ID or recording-ID overlap.
Each input is a 25 fps, 96×96 RGB mouth-crop MP4, with 13–154 frames.
The training split contains approximately 27.78 hours of video.

The source distribution contains 25 fewer training clips than Oxford's count.
The reason has not been independently established; no additional clips were
silently removed for this run. Counts alone do not verify exact official filelist
membership, because the official lists require credentials. This is evaluation
on a processed LRS2 distribution, not an exact reproduction of raw author crops.
See `runs/lrs2-full-v1/dataset-protocol.json` and the pinned source provenance.

The pretrained model's LRS2 training history does not by itself mean it trained
on validation or test. We have not independently audited every pretraining input.
Earlier LRS2 test scores are already known; no test predictions may guide this
run's checkpoint selection, stopping, hyperparameters or decoding.

## Unchanged recipe

- Original pretrained Auto-AVSR SHA256
  `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`.
  Start from these weights with a fresh optimizer, not LRS3-adapted weights.
- Original 250,383,410-parameter architecture, all parameters trainable.
- AdamW lr `1e-5`, betas `(0.9, 0.98)`, weight decay `0.03`.
- Microbatch 1, accumulate 16, correct final partial-group averaging; float32,
  gradient clipping at norm 1; one-epoch warmup and cosine decay over 20 epochs.
- Author hybrid 0.1 CTC + 0.9 attention loss and label smoothing.
- Author random spatial crop to 88×88, AdaptiveTimeMask(10,25), normalization
  mean 0.421/std 0.165. Center crop for evaluation. Existing float RGB-to-gray
  coefficients .2989/.587/.114; no extra quantization of the LRS2 video inputs.
- Frozen BN running statistics with trainable scale/bias.
- MPS visual frontend/encoder, CPU decoder/CTC with connected gradients.
- Maximum 20 epochs, minimum 3, patience 4, no wall-clock cutoff.
- Lowest validation WER chooses the checkpoint; ties keep the earlier candidate.
  The original model remains an eligible fallback.
- Fixed beam 40, CTC weight .1, no external LM or length bonus.
- Evaluate original and selected models on test only after writing selection.json.

The dataset size changes the number of steps per epoch, as expected; the
optimizer and epoch-based learning-rate formula are otherwise identical.
`recipe-equivalence-check.json` confirms matching recipe settings and identical
core methods, exact evaluation-adapter equivalence, all references CTC-feasible,
finite gradients throughout the model and representative weight changes. Its
single training-only feasibility update was discarded. Original weights remain
unchanged. No test/validation clip entered that gradient check.

## Live status and safe recovery

- Dashboard: http://127.0.0.1:8766/?run=lrs2 (LRS3 remains selectable).
- Reports: `training/runs/lrs2-full-v1/` under this project.
- Workspace log: `work/lrs2-only/experiment.log`.
- Runtime: `/private/tmp/grid-clean-env/bin/python`.
- Large active checkpoints: `/private/tmp/autoavsr-lrs2-full-v1` outside iCloud.
- Managed caffeinate launch session: 64206; inspect actual process state instead
  of assuming a session identifier proves liveness.

The job holds a run-specific exclusive lock. Never duplicate healthy work.
Eight workers verify source file hashes, FPS, shape/frame counts and decoded
pixel hashes across splits. Baseline validation follows verification. Atomic
`last.pt` includes the model, optimizer, config, progress and history. Writes occur
every 100 updates and at epoch boundaries, and stop below 8 GiB free disk space.
Resume only with unchanged manifest/config/source from that atomic checkpoint.
Do not edit the running trainer or change the dataset based on model results.

For recovery only, from the project directory:

```bash
MPLCONFIGDIR=/private/tmp/lipnet-exact-mpl /private/tmp/grid-clean-env/bin/python -u -m training.lrs2_finetune
```

## LRS3 preservation

All four existing v1/v2 checkpoint files (two best-model files and two resumable
latest checkpoints) were hash-verified and moved into persistent local storage:
`/Users/taeyeonpaik/Library/Application Support/LipReader/preserved-lrs3-2026-09-20`.
Original `/private/tmp` paths are symlinks to these preserved directories, so the
unchanged LRS3 resume configuration continues to resolve them. A verified ZIP in
that directory contains source, reports, manifests, logs and documents. Dataset
media remains in its existing location and has not been deleted.

The latest recoverable LRS3 state is epoch 3, offset 1,440, global update 3,900.
The best completed candidate is epoch 2 with validation WER 24.9777% on 296 clips.
Its final 1,025-clip test evaluation has not run. See
`runs/lrs3-only-v2/pivot-preservation.json` and the durable `PRESERVATION.json`.
Older intermediate last.pt snapshots had been overwritten before this request;
all checkpoint files that existed at the pivot were preserved, not every past
optimizer update. No LRS3 training should resume without the user's request.

## Completion review

**Current execution update (September 20):** the user subsequently authorized
Lightning GPU tests and batching optimizations. Cloud training now has a separate
`training.lrs2_batched` implementation: microbatch 4, accumulation 4, effective
batch 16, two CPU workers, length buckets, and padding-safe Conformer execution.
Original architecture/optimizer/loss/splits/selection remain. CPU correctness and
actual pretrained inference checks passed; cloud upload and CUDA throughput are
pending while Studio is stopped. See workspace `work/lightning/BATCHING.md`.
Earlier Mac-only and temporary-path descriptions are historical; preserved LRS3
files are durable, and new cloud checkpoints use persistent Studio storage.
No automatic paid training launch is authorized by the implementation request.

Independently recompute WER, CER, exact-sentence accuracy and per-recording paired
uncertainty from saved predictions. Verify checkpoint changes, original SHA,
fresh strict reload predictions, selection before test and split membership.
Compare against the matched original beam-40 baseline, not the old beam-5 score.
Keep negative findings and do not claim sub-15% WER or arbitrary-webcam accuracy.
Only activate an eligible candidate after independent review and actual upload
and recorded-webcam runtime checks; preserve LipNet and original Auto-AVSR.
Update local review/start documents, resume placeholders and compact source/report
archive with actual results. Exclude raw media and optimizer checkpoints from the
distributable archive. Mac only; no deployment, spending or remote compute.

## Speed investigation — September 20

At user request, the process was stopped after atomic update 100 (1,600 training
clips) for isolated training-only execution benchmarks. All scratch gradients and
optimizer updates were discarded. Reports and scripts are in `work/lrs2-speed/`.
CPU thread counts 4/1/2 and MPS decoder execution were tested on the same 16
length-stratified training clips. Small forward/backward improvements did not
establish a substantial end-to-end gain. A repeated complete-update comparison
(including accumulation, clipping and AdamW) measured median 9.87 seconds with
four CPU threads versus 10.27 seconds with one. The original four-thread CPU
decoder layout was retained; no mixed precision, batching, data reduction,
freezing, new loss, learning-rate change or remote compute was introduced.

Restart verification now checks every video's SHA256 against the same manifest
but reuses the already-completed full decoding and cross-split pixel checks.
`decoded-verification-proof.json` pins the original report, config, manifest,
input code and archived trainer. Identical source bytes preserve those earlier
checks. This reduces restart overhead, not the mathematical training workload.
`execution-optimization.json` records the atomic checkpoint-config migration,
with every model and optimizer tensor and progress verified unchanged. The
untouched backup is `/private/tmp/autoavsr-lrs2-full-v1/before-speedup.pt`.
The trainer source/config fingerprint changes only for this documented execution
and verification handling. Restore from current last.pt, not the backup under
new config. No substantial training speedup is claimed from these benchmarks.
