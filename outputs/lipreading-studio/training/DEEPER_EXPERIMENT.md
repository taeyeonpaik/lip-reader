# Deeper fine-tuning: experiment 2

## Completed deeper fine-tuning — September 17

Training and evaluation finished at **14:25 EDT**: **10 full epochs, 15,000 optimizer updates**, and 55,080,306 trainable parameters on 12,000 LRS2 training clips. The final two visual-encoder and final two decoder blocks were updated along with their normalization and output layers. Training loss fell from 4.336 to 2.567.

| Measure | Original pretrained | Best candidate (epoch 7) |
| --- | ---: | ---: |
| Validation word error (500 clips) | 21.40% | 21.06% |
| Validation character error | 13.76% | 13.50% |
| Test word error (1,243 clips) | 16.19% | 16.25% |
| Test word errors / 6,660 reference words | 1,078 | 1,082 |
| Test character error | 10.24% | 10.24% |
| Exact test transcripts | 728 | 739 |
| Failed test clips | 0 | 0 |

**Outcome:** validation WER improved by 0.34 percentage points, below the predefined 0.50-point requirement. Final test WER worsened by 0.06 points (four additional word errors). More exact sentences and a negligible character-error reduction do not establish an overall word-recognition improvement. The app therefore retains the original pretrained weights.

This was real deeper fine-tuning; the saved checkpoint contains changes in every targeted module. Independently recomputed metrics match all evaluation reports, and reloading the best checkpoint reproduced four recorded validation predictions exactly. Both the best and final checkpoints are retained. The original checkpoint hash is unchanged.

The goal of measurably better test word recognition **has not been achieved**. The large training-loss reduction with nearly flat validation error is consistent with fitting this training subset more closely, without a useful generalization gain. No further experiment is running. No website has been deployed.

These scores use prepared LRS2 mouth crops, not arbitrary webcam videos. Auto-AVSR already trained on LRS2, and the test baseline was known from experiment 1. They do not establish new-speaker performance or 83.8% accuracy on personal recordings.

## What changes

The first experiment updated only 7.8 million output-layer parameters and did not improve validation WER. This experiment updates **55,080,306 parameters**: the last two Conformer visual-encoder blocks, the last two attention-decoder blocks, their final normalization layers, and both output projections. The visual frontend and preceding blocks remain frozen. Only frozen-prefix features are cached; trainable visual blocks are recomputed with gradients.

- Data: **12,000 training / 500 validation / 1,243 test clips**, with real LRS2 transcripts.
- Additional public training shards: 4, 7, 15, 19, supplementing shards 0 and 11.
- Validation and test IDs, video hashes, and references are identical to experiment 1.
- No duplicate IDs, identical videos across splits, or overlapping recording IDs.
- AdamW, encoder peak learning rate 1e-5, decoder peak learning rate 1e-5/3, weight decay .01.
- Microbatch four, effective batch eight, gradient clipping at one.
- One epoch of linear warmup, then cosine decay to 10% of peak over ten epochs.
- Up to ten epochs, at least six, then early stopping after four epochs without an eligible improvement.
- Original 0.1 CTC + 0.9 label-smoothed attention objective; dropout in unfrozen blocks; batch-normalization running statistics frozen; 30% chance of masking four feature frames during training.
- Same beam width five for baseline and candidate evaluation.
- Original weights are preserved.

## Selection fixed before training

Choose the lowest validation word error rate among epochs with character error no worse than baseline and no evaluation failures. Require at least **0.5 percentage point** lower validation WER to qualify. Write selection before final test evaluation. Evaluate both original and best candidate on the same final test set, and report both, including a regression if one occurs. Do not automatically activate a candidate merely because training completed.

The previous baseline was 21.40% validation WER and 16.19% test WER. The test baseline is already known; this is not a newly untouched benchmark. The pretrained model already trained on LRS2. These prepared-mouth-crop evaluations do not establish performance on arbitrary new speakers or the full webcam tracking pipeline.

## Verification before the full run

- Frozen-prefix plus trainable-suffix reconstruction exactly matched full-encoder features for a checked example (maximum absolute difference zero).
- Real-label diagnostic updates produced finite gradients and changed both visual and decoder block weights.
- Frozen parameter spot checks remained exactly unchanged.
- Three four-clip batch checks completed with finite losses and gradients on Apple MPS plus CPU.
- Evidence: `../validation/deeper-feasibility.json` and `../validation/deeper-batch-feasibility.json`. These are engineering checks, not accuracy measurements.

## Follow progress

Open the local app's **Model & training** tab. The new experiment appears above experiment 1 and refreshes every ten seconds.

Detailed artifacts are under `runs/lrs2-deeper-v2/`: `config.json`, `progress.json`, per-epoch predictions and metrics, `history.json`, `last.pt` (optimizer and weights), `best.pt` if an eligible improvement occurs, and finally `result.json`.

The sequence is baseline validation → frozen feature preparation → training and validation for each epoch → final evaluation. A running progress file is not evidence of completion. Completion requires `result.json` and checkpoint/result review.

## Reproduce or resume locally

From the workspace root, acquire the six pinned training shards and original validation/test files if needed:

```sh
PYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.download \
  --destination work/deeper/data --deeper
PYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.data \
  --downloads work/deeper/data/downloads.json --destination work/deeper/dataset \
  --train-limit 12000 --validation-limit 500
```

Resume the existing run with its original manifest; choose a new run directory for a fresh experiment or a manifest reconstructed at a different absolute path:

```sh
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.deeper \
  --manifest work/deeper/dataset/manifest.json \
  --cache work/deeper/features \
  --run outputs/lipreading-studio/training/runs/lrs2-deeper-v2
```

Do not launch a duplicate process. A run lock prevents simultaneous writers. Recovery repeats an interrupted epoch from its preceding checkpoint and clears that epoch's partial evaluation predictions. The exact configuration is checked before resuming. Logs are at `work/deeper/experiment.log` in the workspace. The app is not deployed.

The experiment and final review are complete. No additional training is scheduled. The monitoring follow-up is paused.
