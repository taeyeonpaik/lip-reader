# Original LipNet fine-tuning — paper-finetune-v1

**Training completed:** six epochs, 1,192 usable training clips, 61.6 minutes of optimization. Epoch four was selected before final-test evaluation. Independent review and activation are recorded in `runs/paper-finetune-v1/independent-review.json` and `models/paper-selection.json`; verification passed and epoch 4 is active with the original fallback.

The reproducible comparison, using identical deterministic spelling ties for both checkpoints, is validation **20.33% → 18.44% WER** and final test **18.00% → 17.25% WER**. Test paired 95% delta interval is **−2.00 to +0.50 percentage points**, so the test gain remains inconclusive. No additional run or epoch selection was performed after testing. See COMPATIBILITY.md for the spelling-order issue found during reload verification. Earlier unchanged reports record the original process-specific test result of 18.08% → 17.17%.

The user requested genuine fine-tuning after the source-faithful baseline was verified. This run starts with rizkiarm's published unseen-speaker checkpoint and optimizes the actual original Keras model2. It keeps the original dlib crop, RGB normalization, GRU defaults, CTC loss and beam-200/spelling decoder.

## Fixed recipe

- 1,200 training clips: 600 each from s3 and s4.
- 150 validation clips from s1; 200 final-test clips from s2.
- Prior source-restoration evaluation clips are excluded. Selection is deterministic with seed 193; references are checked against independent `.align` transcripts and clip IDs. Every source video and alignment is hashed.
- Adam learning rate 1e-5, epsilon 1e-8. Microbatch 2 with eight-step gradient accumulation (effective batch 16).
- Batch-normalization parameters and running statistics are frozen to avoid replacing the author's learned statistics with noisy two-example estimates. All other trainable parameters are optimized. Original spatial dropout remains active.
- Original flip probability 0.5 and temporal jitter 0.05. Training clips are padded to 75 frames with the source's padding routine. Validation/test use the actual original frame count.
- Maximum six epochs and 7,200 seconds of training, checked between epochs. Early stop after two epochs without a new best validation WER once three epochs have completed. The time budget takes precedence over minimum epochs. Preparation/evaluation time is additional.

These are explicit fine-tuning choices for this Mac, not a claim to duplicate the author's original training run. No architecture or decoder sweep is authorized or configured. The restored baseline and original weight files stay intact.

## Evaluation and selection

Compute a fresh baseline on this run's validation and test sets. The earlier 18.50% result uses different clips and is not the comparison denominator for this experiment.

Choose the best epoch using validation only. A candidate qualifies for review only if validation WER drops at least 0.5 percentage points and a paired clip-bootstrap 95% interval for candidate-minus-baseline WER is entirely below zero. This interval is conditional on one validation speaker; it cannot establish general performance across speakers.

Write `selection-before-test.json` before final-test evaluation. Do not select another epoch, change settings or launch another run because of test results. Save all predictions, raw and post-spelling WER, exact sentence counts, weight-change evidence and unchanged-author-weight checks. Recompute metrics independently after completion. Verify candidate weight reloads and upload/recorded-webcam latency before activation. Final test can veto activation for regression but cannot select another candidate. The physical camera remains untested unless actually tested.

The original checkpoint's upstream scenario declares s1/s2 held out. Locally, neither enters gradient updates. Both speakers have appeared in previous project evaluations, so this is a fixed held-out evaluation for this run, not a wholly untouched research benchmark. Training on s3/s4 alone also limits speaker diversity. A gain is an experiment outcome, not a promised sub-10% result.

## Storage, process and recovery

### Preprocessing recovery, before epoch one

The first preparation attempt stopped at `s3/bgbn9a.mpg`: the original dlib detector returned full frames instead of mouth crops. The original algorithm is preserved. Invalid training crops are now logged and excluded without replacement; the original selected manifest remains intact and an effective manifest records the usable training rows. Validation/test keep every selected clip, counting preprocessing failures as empty predictions (six word errors for a GRID sentence). More than 5% invalid training crops stops the run for review. Valid caches are reused. Gradient accumulation weights a partial final microbatch by its actual example count. No model, decoder, learning rate, split membership or stopping rule was retuned.

`preprocessing-recovery-check.json` records a regression check on the actual failed clip, cache reuse and evaluation failure counting. `preparation-report.json` lists every rejected crop and `effective-manifest.json` records the eventual training count. No epoch or trained checkpoint was lost in this interruption.

`runs/paper-finetune-v1/` holds the fixed config, hashed manifest, progress, predictions, one resumable optimizer checkpoint and best candidate weights. Original RGB crops and recovered source data live under `/private/tmp/lipnet-paper-finetune-v1/`, outside iCloud Documents. This temporary data can be rebuilt from verified sources if cleared.

Run using `/private/tmp/lipnet-exact-tf/bin/python`, from the parent workspace:

```sh
PYTHONPATH=outputs/lipnet-exact /private/tmp/lipnet-exact-tf/bin/python -u outputs/lipnet-exact/finetune.py
```

The launch recovery script waits for the known s4 download to complete, verifies the original MD5, extracts only s4 video, and then starts this command under `caffeinate`. A file lock rejects duplicate fine-tuning processes. Cache metadata must match the video hash and preprocessing code hash. Config changes are rejected on resume. The single saved checkpoint restores model and optimizer after a completed epoch; interrupted epochs are rerun from the last committed checkpoint. Random runtime behavior is not guaranteed bit-identical across process restarts. A checkpoint/state transaction mismatch stops for review rather than guessing.

Logs: `work/lipnet-exact/finetune.log`, `finetune-smoke.log`, `s4-recovery.log`.
Progress: `runs/paper-finetune-v1/progress.json`.
Completion: `runs/paper-finetune-v1/result.json`.

## Finish checklist

1. Independently recompute validation and final-test metrics from saved predictions; verify speaker disjointness, labels and candidate selection timing.
2. Verify published weights unchanged, candidate tensors changed, saved/reloaded predictions equal and checkpoint token compatibility.
3. Compare candidate and baseline runtime on identical upload and recorded-webcam inputs. Integrate a candidate only after review supports both improvement and runtime. Keep the author fallback and practice prompts out of recognition.
4. Record positive, negative or inconclusive findings in README, REVIEW, START_HERE and provenance. Clearly distinguish completed fine-tuning from the preceding restoration-only stage.
5. Exclude optimizer resume files and full datasets from the ZIP. Rebuild and verify it, then remove the completion heartbeat. Retain original source, author weights and compact experiment evidence.

The project cleanup is recorded in `outputs/STORAGE_CLEANUP.json`: over 41 GiB of obsolete caches, duplicate datasets, environments, old ZIPs and rejected optimizer checkpoints were removed. Original checkpoints, current GRID data and historical reports were preserved.

## Completion

All evaluation, reload, weight-integrity, upload/recorded-worker runtime and activation checks passed. `REVIEW.md` records the final stable comparison and its uncertainty. The selected checkpoint is published locally through `models/paper-selection.json`; the source author weights are preserved. Full datasets and optimizer resume files are excluded from the reviewed ZIP. No additional training campaign was launched.
