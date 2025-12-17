# Combined LRS2/LRS3 fine-tuning — experiment 4

## Status

## Completed mixed-data fine-tuning — September 18

Experiment 4 finished at **00:47 EDT**, after approximately **7 hours 41 minutes**
including baseline evaluation, feature preparation, training and final evaluation.
It completed **three full epochs, 14,304 optimizer updates**, and **228,846 clip
presentations** using 76,282 training clips (45,814 LRS2 + 30,468 verified LRS3).
The preset early-stopping rule ended the run because no epoch qualified; six
was the maximum, not a promise to continue after the stopping condition.

| Validation measure | Pretrained, beam 20 | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: | ---: |
| LRS2 WER, 1,082 clips | 22.883% | 22.960% | 23.023% | 22.921% |
| LRS3 WER, 1,488 clips | 8.744% | 8.577% | 8.421% | 8.380% |
| Mean of dataset WERs | 15.814% | 15.768% | 15.722% | 15.651% |
| Training loss | — | 4.024 | 3.837 | 3.733 |

**Outcome:** the final epoch reduced LRS3 word errors by 63 (1,515 → 1,452), but
added three LRS2 errors (1,800 → 1,803). Mean WER improved by **0.163 percentage
points**, below the required 0.50-point gain, and every epoch regressed on LRS2
WER. No adapted checkpoint qualified. The app keeps the original pretrained
weights. The final trained weights and optimizer state are saved in `last.pt`;
there is no eligible `best.pt` for this run.

All five mixed-data evaluation reports and all 14 decoding reports were
independently rescored. Saved weights changed in every targeted module, and
reloading the final checkpoint reproduced four recorded validation predictions
(two from each dataset). The original checkpoint hash is unchanged.

No adapted test evaluation was run because no validation candidate qualified.
The **16.14%** LRS2 test WER in experiment 4 is the unchanged pretrained model
with beam 20. The active app keeps faster beam 5, whose measured LRS2 test WER
is **16.19%**. The small decoding difference must not be attributed to training.

LRS3 validation comes from the approximately 30-hour trainval release, held out
of local fine-tuning. The base model already trained on LRS3. Its sub-10% WER on
this partition is neither a new official benchmark nor evidence of sub-10% WER
on personal webcam recordings. The goal of a robust overall improvement over
the pretrained model remains unmet. No additional training campaign is running.


## Evidence motivating the run

The previous 12,000-clip LRS2 run decreased training loss substantially without
improving test WER. Decoding changes produced only tiny gains: LRS2 test WER
16.19% → 16.14%; extra LRS2 validation 24.44% → 24.42%; LRS3 session-validation
8.915% → 8.897%. On the extra LRS2 validation subset, character error worsened
slightly. These results do not establish a substantial generalization gain.

The LRS3 baseline was already below 10% on this locally held-out trainval subset.
That must not be presented as a new achievement: the original checkpoint already
trained on LRS3, and this is not the official LRS3 test benchmark.

This experiment tests a larger and more varied training set with a lower
learning rate and larger effective batch. It starts from the original author
weights, not the previously overfit candidate. No claim that more data or this
recipe guarantees improvement is warranted.

## Fixed recipe

- **76,282 training clips:** 45,814 LRS2 main + 30,468 verified LRS3 trainval.
- **2,570 validation clips:** all 1,082 LRS2 validation + 1,488 held-out LRS3
  recording-session clips. No validation clips enter gradient updates.
- **1,243 LRS2 test clips:** evaluate after validation selection; the baseline
  has been examined previously. No official LRS3 test is included.
- **55,080,306 trainable parameters:** last two visual-encoder blocks, last two
  decoder blocks, their final normalization layers and output projections.
- Frozen frontend and first ten encoder blocks are cached in float32. LRS2 RGB
  and LRS3 grayscale preprocessing are handled separately and checked.
- AdamW; peak visual learning rate **3e-6**, decoder **1e-6**; weight decay .01.
- Effective batch **16**; microbatch 4; gradient clipping at 1.
- One epoch warmup, cosine decay to 10% peak; seed 44.
- Training-only four-frame feature masking with probability .15; dropout in
  unfrozen blocks; batch-normalization running statistics frozen.
- Up to **6 full epochs**, at least 3, stopping after 2 epochs without an
  eligible validation improvement. Approximately 4,768 optimizer updates/epoch.
- Fixed **beam 20, no external language model** for both baseline and candidate.
  This controls decoding when measuring the effect of weight updates.

## Selection and reporting

Compute corpus WER/CER separately for LRS2 and LRS3. Rank candidates by the
arithmetic mean of the two WERs, giving the datasets equal weight regardless of
reference-word counts. An eligible candidate must not increase WER or CER on
either dataset and must have no evaluation failures. Require at least a 0.5
percentage-point reduction in mean WER before recommending adapted weights.

The full per-dataset scores, pooled scores, training loss, predictions and
checkpoints are retained. Test results do not choose the epoch. Do not
silently activate a candidate at completion or make new-speaker claims.

## Data verification and resources

The independent author label list verified 31,956 LRS3 IDs, frame counts and
transcripts. One clip missing from that reference was quarantined; source media
is preserved. Use **work/corpora/verified-manifest.json**. The original combined
manifest contains that excluded clip and must not be used for training.

Preflight checked all 80,095 retained rows: no CTC alignment-length failures;
maximum length 155 frames. Estimated additional feature cache: 13.12 GiB, with
29.7 GiB free at preflight. Cache generation stops before free space falls below
8 GiB. Reuse `work/deeper/features` from the prior experiment.

`validation/mixed-training-feasibility.json` confirms finite gradients through a
real batch containing both datasets, changed target modules, unchanged frozen
parameter, and per-dataset metric aggregation. It is not an accuracy result.

## Progress and recovery

Run directory: `training/runs/lrs2-lrs3-mixed-v4/`.
Workspace log: `work/mixed/experiment.log`.

The order is baseline evaluation of both datasets → preparation of frozen
features → training/validation → final selected-checkpoint evaluation. The
baseline stage itself takes time; it is not an optimizer update.

```sh
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.deeper \
  --manifest work/corpora/verified-manifest.json --cache work/deeper/features \
  --run outputs/lipreading-studio/training/runs/lrs2-lrs3-mixed-v4 \
  --balanced-datasets --beam-size 20 --lr 3e-6 \
  --accumulation 16 --microbatch 4 --epochs 6 --min-epochs 3 --patience 2 \
  --mask-probability .15 --seed 44
```

Do not duplicate a running process. The run has an exclusive lock. Recovery
uses the same configuration and last complete epoch; an interrupted epoch is
replayed. `last.pt` contains weights and optimizer; `best.pt` contains an eligible
best checkpoint if one exists. `result.json` marks completion, followed by
independent checkpoint and metric review. The optional macro-WER field is named
`best_validation_macro_wer` in the final selection report.

## Final review

Checkpoint and independent metric verification passed; see
`validation/mixed-checkpoint-and-metrics.json`. The app retains original weights.
The trained final checkpoint remains available for reproducibility, with no
adapted test result because none qualified on validation. Review documents and
the archive include the completed experiment and its negative overall outcome.
No further fine-tuning or deployment was started.

## Final runtime checks

On the included eight-second video, two uploads including normalization, face
tracking and recognition took **4.47 / 3.03 seconds with beam 5**, versus
**4.72 / 4.67 seconds with beam 20**. Model loading is excluded.
Two paced eight-second webcam windows per decoder processed in **2.13 / 2.10
seconds with beam 5** and **3.82 / 4.41 seconds with beam 20**, with zero skipped
windows. This is a recorded-video throughput check, not a physical-camera test
or a guarantee for other speakers. Timing details are in
`validation/decoding-throughput.json`. The faster beam-5 default is retained.


## Final verification

All **23 regression tests passed**. The local WebRTC transport test received 200
video frames with no audio tracks or skipped segments. The completed results
panel was checked in the browser. Checkpoint, independent metric, runtime and
UI evidence are retained in `validation/`. Physical-camera testing remains a
manual review step. No website was deployed.
