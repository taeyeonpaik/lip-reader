# Lip Reading Studio — review report

## Official LRS3 test baseline — September 19

The unchanged pretrained Auto-AVSR checkpoint achieved **27.45% WER** (2,715 / 9,890 word errors; 0 inference failures) on all **1,321** verified test-list utterances from the public processed LRS3 test mirror, using the current app's beam-5 decoder. The complete transcript/frame-count multiset matches independent author test labels, with no IDs or recordings shared with our verified trainval data. Six repeated-phrase ID assignments remain ambiguous and are retained without exclusions. Mirror crop provenance is limited; this is not a raw-video byte verification or exact reproduction of the authors' benchmark pipeline. All metrics were independently recomputed and fixed sample predictions reproduced after reload. This replaces local trainval scores as our test baseline; no weights changed. [Full review](training/runs/lrs3-official-test-v1/REVIEW.md).


## VideoMAE transfer learning — completed September 18

**Outcome: unsuccessful transcription model.** The bounded Mac experiment
completed 32 epochs over 5,000 clips, selecting epoch 24 by validation CER.
Feature extraction, training and final evaluation finished in 79.6 minutes at
18:42 EDT. Training stopped after eight epochs without a better validation score.
The frozen generic Kinetics VideoMAE encoder plus newly trained BiGRU/CTC head
produced mostly incorrect character sequences. No complete evaluation sentence
was recognized correctly. Auto-AVSR remains the default working model.

| Evaluation (200 clips per row) | VideoMAE WER | Auto-AVSR WER, same clips |
| --- | ---: | ---: |
| LRS2 validation | 112.99% | 23.08% |
| LRS3 local validation | 115.71% | 10.38% |
| LRS2 official-test subset | 117.18% | 16.41% |
| LRS3 local trainval audit | 115.34% | 8.62% |

The mean of per-dataset heldout WERs is **116.26%** for VideoMAE versus **12.51%**
for Auto-AVSR. Pooling all heldout words instead gives 115.91% versus 11.03%; these
are different aggregations. WER can exceed 100% because insertions count as errors.
The LRS3 audit is **not the official LRS3 benchmark**. These corpora appeared in
previous project experiments, and the Auto-AVSR pretrained model used LRS2/LRS3.

Training loss fell from 3.020 to 2.093, but lower training loss did not establish
useful word recognition. On validation, removing all visual features worsened
macro character error from 71.38% to 79.70%, while word error fell from 114.35%
to 101.46%. This indicates limited visual signal at the character level, not
successful transcription. This experiment does not isolate whether freezing,
feature pooling, dataset size or the decoder caused the poor outcome.

Thirty head tensors changed; the generic encoder and original Auto-AVSR weights
are unchanged. All 38 prediction reports were independently recomputed; published
weights exactly match the selected checkpoint and four raw-crop reload predictions
matched. The earlier full regression suite passed 31 checks.

An eight-second recorded example took 2.13 seconds for face tracking/recognition
and 1.37 seconds through the recorded-webcam worker after capture. Conversion,
model loading and the eight-second capture window are additional. These timings
are individual runs, not a latency guarantee. **No physical webcam was tested.**
The model remains selectable only as **VideoMAE · experimental**, with an explicit
failed-experiment warning, for reviewing the implementation. It is not recommended
for transcription. No extra training, deployment or new model campaign was started.

Full details: [experiment recipe](training/VIDEOMAE_EXPERIMENT.md),
`training/runs/videomae-v6/result.json`, `validation/videomae-completion.json`.
The reviewed ZIP includes code, weights, reports and dataset provenance; raw
LRS2/LRS3 media and feature caches are excluded.



## Ready to review

Open [the local app](http://127.0.0.1:8502) or double-click **Start Lip Reading.command**.
The website remains undeployed. The app supports video uploads, live webcam
transcription, and TXT/SRT/JSON export. It processes images locally and does not
use the microphone or uploaded audio for recognition.

This is a pretrained **Auto-AVSR application** for natural English. It uses a different architecture from the original LipNet tutorial, as agreed.

## Final MWER pilot completed — September 18

The final bounded attempt finished at **07:59 EDT**, after approximately
**76 minutes** of diagnosis, training and evaluation. Both conventional training
and MWER completed **two full epochs and 250 updates each** on the same 1,000
training clips, with identical initialization and data order. Each arm updated
26,670,450 decoder/CTC parameters; the visual encoder stayed frozen.

### Validation results (400 clips, 200 per dataset)

| Model | LRS2 WER | LRS3 WER | Mean of dataset WERs | Total word errors |
| --- | ---: | ---: | ---: | ---: |
| Pretrained, beam 20 | 25.83% | 10.99% | 18.41% | 638 |
| Conventional, epoch 1 | 25.83% | 10.76% | 18.29% | 633 |
| Conventional, epoch 2 | 25.51% | 10.58% | 18.04% | 624 |
| MWER, epoch 1 | 25.83% | 10.85% | 18.34% | 635 |
| MWER, epoch 2 | 25.57% | 10.67% | 18.12% | 627 |

**Outcome:** MWER corrected 11 word errors overall, reducing mean WER by **0.289
percentage points**. The matched conventional-training control corrected 14,
reducing mean WER by **0.367 points**. MWER did not outperform the control.
Neither met the predefined 0.50-point improvement requirement. No candidate was
selected, and the conditional additional audit/test evaluations were not run.
These results do not establish an improved test model.

The diagnostic found a better candidate among the top ten token sequences for
160 of 400 clips. Choosing candidates with the reference answers gave **11.32%
mean WER**, versus 18.41% for actual top-ranked predictions. This is an oracle
score: it uses the answers and is NOT deployable model accuracy. The diagnostic
supported trying ranking-focused training, but the pilot captured little of that
potential. The result does not prove that all possible MWER recipes would fail.

Conventional training loss decreased from 3.509 to 3.206. MWER's centered
objective decreased from −0.249 to −0.328; negative values are expected because
candidate error counts are centered. Loss values across these different
objectives are not directly comparable.

The app retains the original pretrained weights and fast beam-5 decoding.
Its previously measured **16.19% LRS2 test WER** is unchanged. All pilot comparisons
use beam 20 and a different validation subset, so their numbers must not be
compared directly with that test score or the larger experiment-4 validation.

**This completed the Auto-AVSR training attempts.** A separate VideoMAE transfer-learning experiment was subsequently authorized; its status is above. No deployment is planned. The app remains usable locally,
and all candidate checkpoints, split provenance, predictions and experiment
code are retained for review. Dataset media is excluded from the app archive.

## MWER verification and uncertainty

All five evaluation reports and the candidate-oracle score were independently
recomputed. Reloading all four epoch checkpoints reproduced eight recorded
predictions (one per dataset per checkpoint). Every targeted module changed,
and the original checkpoint hash remained intact.

A paired bootstrap over recording sessions (2,000 repetitions, stratified by
dataset) estimated these 95% intervals for **mean WER reduction**, in percentage
points:

- Conventional versus pretrained: **+0.053 to +0.743** (estimate +0.367).
- MWER versus pretrained: **+0.017 to +0.600** (estimate +0.289).
- MWER versus conventional: **−0.405 to +0.203** (estimate −0.078).

The first two intervals support small gains on this validation sample. They do
not remove bias from choosing the better epoch on the same validation data, or
establish unseen-speaker/test improvement. The comparison between methods gives
no evidence that MWER is superior. The predefined minimum gain was not met, so
we preserve the original app model and finish the attempt as requested.

All 26 regression tests passed before the run; the active recognition pipeline
is unchanged. Previously verified upload/WebRTC throughput therefore remains the
relevant runtime evidence. No candidate was activated or deployed.

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
the pretrained model remains unmet. Experiment 4 is finished; see the latest status above for subsequent work.

## Final runtime checks

On the included eight-second video, two uploads including normalization, face
tracking and recognition took **4.47 / 3.03 seconds with beam 5**, versus
**4.72 / 4.67 seconds with beam 20**. Model loading is excluded.
Two paced eight-second webcam windows per decoder processed in **2.13 / 2.10
seconds with beam 5** and **3.82 / 4.41 seconds with beam 20**, with zero skipped
windows. This is a recorded-video throughput check, not a physical-camera test
or a guarantee for other speakers. Timing details are in
`validation/decoding-throughput.json`. The faster beam-5 default is retained.

## Completed decoding and data expansion — experiment 3

Wider search and the authors' external English language model are implemented
and evaluated. The external-LM finalist failed the predefined character-error
condition. Beam 20 without LM was selected on 500 validation clips before test
and cross-dataset audits. Compared with beam 5, it removed three word errors on
1,243 LRS2 test clips, and one error on each additional validation subset. Extra
LRS2 character error worsened. Prepared-crop test processing averaged 0.31 seconds
with beam 5 and 0.48 seconds with beam 20. These small differences do not justify
replacing the faster app default. Both configurations remain reproducible in code.

Combined data acquisition and independent label verification are complete.
One LRS3 clip missing from the author label list was excluded; verified manifests
contain 76,282 training, 2,570 validation and 1,243 test clips. Dataset provenance,
IDs and hashes are published with this project; dataset media stays outside the
application package. See `training/combined-data-provenance.json`.

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

The goal of measurably better test word recognition **has not been achieved**. The large training-loss reduction with nearly flat validation error is consistent with fitting this training subset more closely, without a useful generalization gain. Experiment 4 also completed; its results are described above. No website has been deployed.

These scores use prepared LRS2 mouth crops, not arbitrary webcam videos. Auto-AVSR already trained on LRS2, and the test baseline was known from experiment 1. They do not establish new-speaker performance or 83.8% accuracy on personal recordings.

## Earlier output-layer experiment (experiment 1)

The local adaptation experiment completed on the Mac M1 Pro in **39.5 minutes**,
finishing at 02:56 EDT on September 17. It used real LRS2 labels and prepared
mouth-crop videos: 4,000 training clips, 500 validation clips, and 1,243 test clips.
Only the two output projections were updated: 7,765,362 of 250,383,410 parameters.

Three full epochs completed, totaling **1,500 optimizer updates**. Early stopping
ended the run after three epochs without improved validation word error. The
original pretrained model remains the app's selected model. The adapted
checkpoints, optimizer state, history, source hashes and individual predictions
are retained for inspection and reproducibility.

| Model / epoch | Validation word error | Validation character error |
| --- | ---: | ---: |
| Original pretrained | 21.40% | 13.76% |
| Adapted epoch 1 | 21.40% | 13.75% |
| Adapted epoch 2 | 21.54% | 13.88% |
| Adapted epoch 3 | 21.48% | 13.82% |

Training loss decreased from 4.040 to 3.953, but this did not translate to better
validation recognition. **Fine-tuning completed; an accuracy improvement was not
established.** Keeping the stronger baseline was the predefined fallback.

### Final test result: selected pretrained model

| Measure | Result |
| --- | ---: |
| Test clips | 1,243 |
| Reference words | 6,660 |
| Word errors | 1,078 |
| **Word error rate** | **16.19%** |
| Character error rate | 10.24% |
| Exact transcripts | 728 / 1,243 (58.57%) |
| Failed examples | 0 |

Word error includes substitutions, omissions and extra words. It is not sentence
accuracy. This result uses prepared LRS2 mouth crops and does not include face
tracking. It does **not** establish 83.8% accuracy on arbitrary personal videos.
The base model already trained on LRS2; speaker novelty has not been established.
The test split was not used to choose the model or tune its parameters.

## Product verification

- All **14 regression tests passed**, including audio removal, real mouth
  alignment, invalid-video rejection, chunk timing, webcam buffering, stopping,
  corpus error calculations, split leakage checks and adapter serialization.
- The actual browser upload flow produced a transcript for the included eight-
  second speaking-person example. TXT, SRT and JSON controls rendered correctly.
  The latest browser run took 9.1 seconds from clicking Read, including conversion;
  speed varies with machine load.
- Two local WebRTC peers delivered 200 recorded-video frames through the real
  asynchronous webcam processor and recognizer: no audio track, no skipped
  segments, successful final stop. Total time was 13.24 seconds, including capture.
- The completed training page displays the chosen model, test error and history.
- Physical camera capture has **not** been exercised: the automated test uses a
  recorded video. Browser/macOS camera permission and your own speech remain a
  manual review step.

The included public example is a functional demonstration, not an independent
accuracy benchmark. Its upload and WebRTC predictions can differ because WebRTC
compression changes the images. Live results arrive after eight-second capture
windows plus recognition time; this is not instantaneous word-by-word captioning.

## Your five-minute review

1. Open the app and run **Try an example**.
2. Upload a 5–15 second video with one well-lit, front-facing English speaker.
3. Read the transcript and export text or subtitles.
4. Open **Live webcam**, grant permission, and speak for at least eight seconds.
   Stop and check the final transcript.
5. Open **Model & training** to inspect the measured results.

For camera trouble, try the localhost URL in Chrome or Safari and check camera
permission. Keep the whole face visible. Only one speaker can be processed at a
time; strong head turns, a covered mouth, tiny faces and unusual lighting can
reduce recognition quality.

## Résumé positioning

**Lip Reading Studio — visual speech recognition application**

- Built a local English lip-reading app with video upload, live WebRTC capture,
  face alignment, visual-only inference and subtitle export using PyTorch,
  MediaPipe and Streamlit.
- Implemented and evaluated two Auto-AVSR fine-tuning experiments, including
  55.1 million trainable parameters on 12,000 LRS2 clips over 10 epochs; compared
  validation and test errors and retained the pretrained baseline when test word
  recognition did not improve.

Credit the pretrained architecture and base weights to the Auto-AVSR authors.
Do not describe this as a model trained from scratch, a LipNet reproduction, or
proven high accuracy on arbitrary webcam speech.

## Evidence and source

- [Deeper experiment results](training/runs/lrs2-deeper-v2/result.json)
- [Deeper training history](training/runs/lrs2-deeper-v2/history.json)
- [Checkpoint and metric verification](validation/deeper-checkpoint-and-metrics.json)
- [Deeper training recipe](training/DEEPER_EXPERIMENT.md)

- [Experiment result](training/runs/lrs2-heads-v1/result.json)
- [Epoch history](training/runs/lrs2-heads-v1/history.json)
- [Per-clip test predictions](training/runs/lrs2-heads-v1/baseline-test.json)
- [Dataset provenance and split checks](training/data-provenance.json)
- [Training recipe](training/README.md)
- [WebRTC verification](validation/webcam-webrtc-test.json)
- [Regression test output](validation/tests.txt)
- [Application documentation](README.md)
- [Auto-AVSR authors' repository](https://github.com/mpc001/auto_avsr)

## Final verification

All **23 regression tests passed**. The local WebRTC transport test received 200
video frames with no audio tracks or skipped segments. The completed results
panel was checked in the browser. Checkpoint, independent metric, runtime and
UI evidence are retained in `validation/`. Physical-camera testing remains a
manual review step. No website was deployed.
