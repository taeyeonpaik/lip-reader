# LRS3-only full-model fine-tuning

## Purpose and status

The user authorized one straightforward Auto-AVSR fine-tuning run using LRS3
only. The full-model memory/gradient check passed and the managed run has been
launched. Initial full-dataset verification precedes baseline validation and
optimization. The author checkpoint and both existing app models remain unchanged.

This uses the author's original 250,383,410-parameter visual model and standard
hybrid CTC/attention loss. All parameters are trainable, including the visual
frontend, all twelve encoder blocks and all six decoder blocks. There are no
adapters, extra language models, ranking losses or hyperparameter sweeps.

## Dataset choice

On September 19, the Oxford LRS3 download page returned 404; no accessible
original full-face test release was found. The public raw mirror was gated.
Following the user's explicitly authorized fallback, use the already verified
grayscale mouth-crop test mirror. Reconstructing the original crop pipeline is
not a prerequisite for this run.

| Role | Clips | Source |
|---|---:|---|
| Training | 30,468 | Verified LRS3 trainval crops |
| Validation | 1,488 | Separate recording sessions from the same release |
| Test | 1,321 | Complete cropped LRS3 test inventory, independently matched to author labels |

Training/validation together cover approximately 30.07 hours. This is not the
full LRS3 pretrain corpus. The previous fixed recording split is retained. No
recording overlaps any other split, including all possible identities of six
ambiguous test rows. Zero LRS2 samples enter this run. The starting author's
checkpoint retains its original mixed-corpus pretraining history.

Sources:

- Trainval: `TheNHz/ellipsis-lrs3-trainval-autoavsr`, revision
  `a9b5374e19082b3217171be8ece589f35c8a2a71`; independent author labels verified
  previously, one unmatched release clip excluded before this run.
- Test: `mattymchen/lrs3-test`, revision
  `9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a`; see
  `runs/lrs3-official-test-v1/dataset-verification.json` for inventory verification
  and crop-provenance limitations. Raw face-video pixels are not verified.

## Fixed ordinary recipe

- Original checkpoint SHA256:
  `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`.
- AdamW, learning rate `1e-5`, betas `(0.9, 0.98)`, weight decay `0.03`.
- One-epoch linear warmup followed by cosine decay; clip gradient norm to 1.
- Microbatch 1, accumulate 16 utterances, correctly average the final partial
  group. Full float32, standard author dropout and label smoothing.
- Author random spatial crop to 88×88 and AdaptiveTimeMask(10,25) during
  training; center crop at evaluation; normalize with mean 0.421/std 0.165.
- Batch-normalization running statistics stay fixed for the small microbatch;
  its learned scale/bias remain trainable.
- MPS visual frontend/encoder and CPU decoder/CTC, with connected gradients.
- Maximum 20 epochs, minimum 3, early stopping after four epochs without a
  lower validation WER. No wall-clock training cutoff.
- Lowest validation WER selects the checkpoint; ties retain the earlier one.
  Include the original model as the epoch-zero fallback.
- Use the author's ordinary beam width 40, CTC weight 0.1, no LM, no length
  bonus for all validation and final comparison. This is fixed from upstream
  defaults, not chosen through a test-set sweep.

The earlier 27.45% test WER used beam 5. It is not a matched beam-40 baseline.
After selection is final, evaluate both original and selected weights on the
same complete test set with the fixed beam-40 decoder. Do not change training
or decoding based on these test results.

## Running and checking progress

From this project directory:

```bash
MPLCONFIGDIR=/private/tmp/lipnet-exact-mpl /private/tmp/grid-clean-env/bin/python -u -m training.lrs3_finetune
```

Read `training/runs/lrs3-only-v1/progress.json` for current stage, epoch, clips
processed, training loss and measured throughput. `history.json` contains
completed-epoch validation results; `selection.json` is written before test
evaluation. The log is `work/lrs3-only/experiment.log` relative to the workspace.

Large weights and optimizer resumes live at `/private/tmp/autoavsr-lrs3-only-v1`,
outside iCloud. Do not purge this directory or the temporary Python environment
while the run is active. The source crop files are reused without duplication.
Checkpoint writes stop if free storage falls below 8 GiB.

The two-clip feasibility check includes a median-length 74-frame clip and a
maximum-length 155-frame clip. Every parameter had a finite gradient; sampled
weights in the frontend, projection, encoder, decoder and CTC changed after
an optimizer update. The second step took 3.90 seconds with about 3.95 GiB of
MPS driver allocation. These checks are not training results and their weight
updates were discarded.

After the user asked to speed up the run, a more representative benchmark used
16 length-stratified training clips (1,354 frames total) with the actual
microbatch-one, accumulation-16 optimizer schedule. Three updates took 15.30,
10.94 and 9.97 seconds, or 0.96/0.68/0.62 seconds per clip. These benchmark
updates were also discarded. The earlier 15–35 hours/epoch estimate based on
the two-clip smoke check was too pessimistic. Approximately 6–10 hours/epoch
including validation is a more useful provisional range; sustained thermal,
input and checkpoint costs remain to be measured. Reports are in workspace
`work/lrs3-only/speed-check/`.

Startup initially stopped before any gradients because the verifier compared
the trainval manifest's original NPY-member hash to the compressed NPZ-file
hash. This was corrected and separately checked on train, validation and test
examples. `startup-failed-*.json` preserves the initial record. The fixed
verifier checks trainval NPY members, test NPZ files, and common pixel hashes
across all splits. No clips were excluded or substituted to resolve this.

On the user's speed-up request, the job was stopped during verification,
before baseline evaluation or any optimization. Verification now reads each
file once, decompresses its NPY member once, and uses eight I/O workers. All
file/member hashes, dtype/shape checks, pixel hashes and split checks remain.
Results are consumed in the original manifest order. Regression checks confirmed
identical pixel hashes on train/validation/test examples and rejection of
modified pixels. Only the source-code fingerprint changed in the configuration;
`io-optimization.json` records the before/after hashes. Data, seed, model,
precision, microbatch, effective batch, optimizer and selection remain identical.
No extra full dataset copy was created. Verification restarts from the beginning,
with earlier files now already read locally; it is expected to be much faster.

The app heartbeat `finish-lrs3-only-auto-avsr-training` checks hourly, stays
quiet on healthy progress, and handles the completion review. Keep the Mac and
Codex running. `caffeinate -i` prevents idle sleep while the training process
is active; closing the laptop lid can still suspend work.

The process has an exclusive lock. Never start a duplicate. A single atomic
`last.pt` stores model, optimizer, config, completed optimizer steps and history.
Updates are checkpointed every 100 optimizer steps and at epoch boundaries.
An interrupted partial accumulation group is replayed from the last saved
optimizer boundary using deterministic per-update seeds. Resume only with
unchanged config, input checksums, source and weights.

## Completion review

1. Independently recompute every validation/test WER from saved predictions;
   verify manifest membership and zero failures.
2. Confirm selected weights differ if a trained candidate won, original SHA
   stays intact, and fresh strict checkpoint loads reproduce predictions.
3. Report paired test uncertainty clustered by recording sessions. Preserve
   ambiguity of the six test identities rather than assigning guessed sessions.
4. Report validation and test independently, including a negative outcome.
   Local trainval validation is not unseen pretraining data. Cropped test results
   are not certified original-preprocessing benchmarks or webcam accuracy.
5. Only after review consider a strict optional full-checkpoint app selector and
   measured upload/recorded-webcam runtime. Preserve LipNet and the author
   Auto-AVSR fallback. Do not deploy.
6. Update review/start documents and a compact report/source archive; exclude
   datasets and optimizer checkpoints. Keep the best usable weights separately.

No sub-15% WER is promised. This is one fixed training experiment, not an
unbounded search for a favorable test score.
