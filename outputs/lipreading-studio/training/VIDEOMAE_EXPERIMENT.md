# VideoMAE transfer learning — experiment 6

## Scope and status

The user authorized this new experiment on September 18, after ending the
Auto-AVSR adaptation experiments. Build a lip reader from a **generic video
pretrained encoder**, train new transcription layers, and support local uploads
and webcam segments. Mac only; no deployment, paid compute or remote training.

Implementation and complete-pipeline smoke checks passed. The main managed run
started at 17:23 EDT on September 18 (session 15754); heartbeat
`finish-videomae-lip-reader` checks every 30 minutes. Main run status is in
`training/runs/videomae-v6/progress.json`, with detailed output in the workspace's
`work/videomae/experiment.log`. A completed result means training **and** the
script's evaluation/reload/runtime checks finished; a checkpoint alone does not.
Training and evaluation are complete; final results and review follow below.

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

Full details: [experiment recipe](VIDEOMAE_EXPERIMENT.md),
`training/runs/videomae-v6/result.json`, `validation/videomae-completion.json`.
The reviewed ZIP includes code, weights, reports and dataset provenance; raw
LRS2/LRS3 media and feature caches are excluded.

## Existing paired training datasets

Use only `work/corpora/verified-manifest.json`: 80,095 verified clips from LRS2 and
the processed LRS3-TED 30-hour trainval release. LRS3 IDs, transcripts and frame
counts were independently checked against author labels; one unmatched clip was
excluded. This is neither the full 407-hour LRS3 corpus nor its official test set.
No new recordings are needed. The provenance and split manifests in `training/`
identify sources and checks without redistributing dataset media.

A deterministic seed/hash selection takes 2,500 training clips per dataset,
limited to 1–6 seconds and valid CTC alignments: **5,000 training clips**.
Validation has 200 clips per dataset. Final evaluation uses 200 LRS2 official-test
samples and 200 local LRS3 trainval audit samples. LRS3 validation/audit use
separate whole recording sessions. Original train/validation/test recording
sessions are also disjoint. Speaker independence has not been established.
These corpora have appeared in previous project experiments; results are not a
new blind benchmark. No test or audit result controls candidate selection.

## Architecture and provenance

- Author VideoMAE Small encoder, pretrained by masked video reconstruction on
  Kinetics-400 (general human-action videos), 21,879,936 frozen parameters.
- Official source: <https://github.com/MCG-NJU/VideoMAE>, commit
  `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`.
- Official Small 1600-epoch pretraining checkpoint from the author's model zoo:
  <https://drive.google.com/file/d/1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww/view>.
- Only encoder tensors retained; reconstruction decoder removed. Strict loading,
  source/checkpoint checksums and CC-BY-NC-4.0 license are recorded in
  `models/videomae-provenance.json` and `vendor/videomae/NOTICE.md`.
- RGB mouth crops, 25 fps; grayscale LRS3 crops replicated over RGB channels.
  Resize 96×96 to 224×224; ImageNet normalization as in author pretraining code.
  Dense consecutive 16-frame windows differ from the author's sparse action-video
  sampling. Final window repeats its last frame for padding.
- Encoder has 2-frame tubelets, spatial-only pooling preserves eight temporal
  vectors per window. Repeat each feature twice to restore 25 Hz CTC alignment
  slots; this repetition adds no visual information.
- Newly initialized LayerNorm/projection, three-layer bidirectional GRU (256
  units per direction), dropout 0.15, and 39-symbol character CTC output.
  **3,274,279 trainable parameters**, trained on CPU with cached MPS visual features.
- Greedy CTC decoding, no external language model. VideoMAE receives no Auto-AVSR
  weights, features, predictions or audio during training. Auto-AVSR runs only
  afterward as a matched evaluation comparator.

This is transfer learning with a frozen encoder, not training all VideoMAE
parameters from scratch or reproducing LipNet. Freezing the encoder limits how
much it can adapt to subtle mouth movements; good transcription is not guaranteed.

## Fixed bounded recipe

Seed 66; balanced datasets; batch 32; AdamW learning rate 0.001, weight decay
0.0001; gradient clipping 5; random 3-feature-step masking with probability 0.5.
Maximum 40 epochs, patience 8 after at least 12 epochs, or a 7,200-second training
wall budget. The budget includes feature extraction and is checked between
complete epochs; at least one epoch completes. Final evaluation is additional.
Selection: minimum macro validation CER (mean of per-dataset character error
rates), with macro WER breaking ties. No hyperparameter sweep or additional
training campaign is scheduled.

Run from the workspace root using its existing environment:

```sh
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.videomae_experiment \
  > work/videomae/experiment.log 2>&1
```

Check for a live process before starting or resuming. An exclusive file lock
prevents two writers. Resume with this exact recipe: verified feature caches and
`last.pt` restore completed-epoch model, optimizer and deterministic epoch order.
Do not change the saved config or start extra training after completion.
A 5 GiB free-space floor protects the disk. LRS2 hashes cover MP4 bytes; LRS3
hashes cover the embedded `frames.npy` payload, matching the original release.

## Verification and app behavior

Before launch, four unit tests passed (CTC repeats/blank handling, variable-length
batching, finite gradients, independent edit-distance accounting). The author
attention implementation and optimized SDPA agree within float32 tolerance
(maximum difference 1.45e-5). A tiny 8-clip/two-epoch pipeline smoke run completed
metric recomputation, strict head reload and four matching predictions, raw-video
tracking, and recorded frames through the actual webcam worker. This verifies
plumbing, not useful accuracy: the tiny smoke head outputs empty transcripts.

Initial feature timing: warm 2.4-second clip encoded in 0.305 seconds. The smoke
upload path took 39.3 seconds on its first 8-second segment, while a subsequent
recorded webcam segment took 1.47 seconds; cold-start costs matter. Final model
runtime is reported in the completed-results section above. A physical camera has not been tested.

The final script independently recomputes all word/character edit counts from
saved predictions; evaluates a zero-visual-input ablation; compares Auto-AVSR
beam 5 on exactly the same validation and heldout clips; checks trained parameter
changes; verifies strict reload predictions and encoder preservation; measures
upload and recorded-webcam processing. Reports include poor or empty predictions.

Only after these checks, the trained head becomes **VideoMAE · experimental**
in the app's model selector. Auto-AVSR remains the default. Successful engineering
checks do not imply that VideoMAE is more accurate. The Model & training tab shows
both models' measured results, epoch progress, WER and CER. Uploaded transcripts
and webcam sessions are cleared when changing models.

The full regression suite passed 31 checks: 29 in the sandbox and the two
MediaPipe graphics tests when rerun with Mac GPU access. The local app was
inspected in the browser: model selector, upload page and live feature progress
render correctly. See `validation/videomae-integration.json`. The final trained
model selector, failure warning and example inference were reviewed in the app.

## Completion checklist for the follow-up

1. Read logs, config, progress and result; do not duplicate a healthy process.
2. Safely resume only after an interruption with the unchanged recipe/caches.
3. Inspect independently recomputed metrics, ablation and matched comparison;
   report uncertainty/limitations without implying an official LRS3 result.
4. Verify the published checkpoint checksum, app selector and both inference paths.
   Inspect actual app UI; do not claim a physical-camera test from recorded input.
5. Update this document, README, REVIEW and START_HERE with actual findings.
6. Rebuild `outputs/lipreading-studio.zip` using `work/overnight/package.py` after
   adding VideoMAE result/checkpoint/verification assertions. Verify ZIP CRC and
   checksums; include code, weights, reports and provenance, not dataset media.
7. Delete the VideoMAE heartbeat when review and packaging finish. Notify the user
   of completion or a meaningful failure; ordinary healthy progress stays quiet.

## Final review

On September 18, the published checkpoint was confirmed tensor-for-tensor equal
to selected epoch 24, both original encoders retained their recorded hashes,
and all 38 reports were recomputed again. The actual app selector, explicit
failure warning, recorded example inference (8 seconds processed in 3.0 seconds),
and completed-results panel were checked. Auto-AVSR was restored after review.
Documentation and provenance are finalized for packaging; no further training
is scheduled. The finished archive checksum is in `outputs/PACKAGE_VERIFICATION.json`
at workspace level.
