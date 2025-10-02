# Final fine-tuning review — 19 September 2026

**Completed:** six epochs of original rizkiarm LipNet fine-tuning on this Mac. Epoch 4 is the reviewed default for uploads and webcam practice. The original pretrained model remains selectable and is the automatic fallback if checkpoint integrity checks fail. Nothing is deployed.

## Results

Both checkpoints use the same original beam-200 decoder and dictionary scores, with deterministic alphabetical ordering for spelling ties.

| Fixed evaluation | Original pretrained | Fine-tuned epoch 4 | Exact sentences, original → fine-tuned |
|---|---:|---:|---:|
| Validation: 150 s1 clips / 900 words | 20.33% (183 errors) | **18.44% (166 errors)** | 34 → 41 |
| Final test: 200 s2 clips / 1,200 words | 18.00% (216 errors) | **17.25% (207 errors)** | 59 → 60 |

The test improvement is **0.75 percentage points**, or 4.17% relative. The paired clip-bootstrap 95% interval for candidate-minus-baseline test WER is **−2.00 to +0.50 percentage points**, including zero. This is a modest observed improvement, not conclusive evidence of better general accuracy. Validation’s interval is −3.00 to −0.78 points and meets the prespecified activation threshold.

Raw beam output before spelling correction improved from 31.11% to 29.56% WER on validation and 31.33% to 30.42% on test. Post-spelling metrics above are the deployed decoder’s results.

## Training and independent checks

- Fixed selected training set: 1,200 s3/s4 clips. Eight invalid crops were excluded without replacement, leaving **1,192**. One validation and two test preprocessing failures remain in their denominators and each count as six word errors.
- Original model2 architecture, dlib crops, RGB normalization, CTC, augmentation and published initialization; Adam 1e-5, microbatch 2, accumulation 8, frozen batch normalization. Six epochs took **61.6 minutes of optimization**, plus preparation and evaluation.
- Epoch 4 was selected on validation and the decision saved before final testing. No epoch reselection, extra run or tuning against the final test followed.
- Independent word-edit-distance recomputation verified every report. All 1,550 selected source videos and alignments and 140 upstream source/asset hashes were verified; train/validation/test speaker sets are disjoint for this run.
- Twenty weight tensors changed; batch-normalization tensors and original published weight files are unchanged. Fresh baseline and candidate loads reproduced predictions for all 150 validation and 200 test clips.
- Strict checkpoint selection validates source, decoder, review and weight hashes and the 28-class vocabulary. Invalid manifests fall back to the original model. Review and selector checks are included with the package.

### Spelling reproducibility correction

The upstream spelling corrector uses a Python set when choosing equally probable words. Different processes can return different spelling ties despite identical neural predictions. Independent reload verification exposed this issue. We fixed ties alphabetically for both checkpoints, without supplying references or changing the selected epoch. Cross-process checks reproduce all 700 baseline/candidate predictions under two hash seeds.

Unchanged training-process reports contain test WER **18.08% → 17.17%**. The stable reports and this review use **18.00% → 17.25%**. Raw predictions are unchanged. Both versions are preserved, and the stable version is the current app’s decoder. Details: `COMPATIBILITY.md` and `runs/paper-finetune-v1/independent-review.json`.

## Runtime and product

For the same 2.98-second bundled full-face clip, median upload processing took **3.57 seconds** for the author model and **3.53 seconds** for the candidate (two runs each). Recorded-webcam worker processing took **3.51** and **3.43 seconds**, respectively, and upload/worker probabilities matched exactly. These are local CPU observations on one clip, not a broad throughput benchmark.

The app includes original examples, video upload, GRID sentence guidance, random prompts, a checkpoint selector and a three-second webcam take followed by transcription. The displayed prompt never enters recognition. This is capture-then-transcribe, not continuous real-time transcription. No microphone or stored webcam recordings are used. **Physical-camera capture and accuracy remain untested.**

## Limits and project claims

There is only one validation and one final-test speaker. Conditional clip-bootstrap intervals do not estimate uncertainty across new speakers. The upstream repository declares s1/s2 held out from its unseen-speaker scenario, but this local run is not a certified reproduction of the complete published benchmark; these speakers also appeared in earlier project evaluations. No general webcam or sub-10% WER claim is supported.

The earlier source-restoration results (notebook 2.83% on an s1 sample; paper 18.50% on another s1/s2 sample) are preserved in [SOURCE_RESTORATION_REVIEW.md](SOURCE_RESTORATION_REVIEW.md). They use different clips and cannot serve as this fine-tuning experiment’s baseline. The notebook author’s complete training split is unknown.

An accurate résumé description: “Reproduced legacy LipNet implementations, fine-tuned the original CNN–BiGRU model on GRID, and built a local video-upload/webcam-practice app with reproducible evaluation and checkpoint safeguards.” If including accuracy, specify **17.25% WER on a fixed 200-clip, one-speaker test subset**, with the modest improvement and evaluation limits available in this review.

The ZIP includes source, licenses, original and fine-tuned weights, small source demos, and reports. Full datasets and optimizer resume checkpoints are excluded. Prior experiments remain historical. More than 41 GiB of obsolete assets were removed, with the cleanup recorded separately.

## Optional unrestricted-English mode

Auto-AVSR has been restored alongside both LipNet choices in the same local app. It uses the retained author weights, without reactivating rejected fine-tuning candidates. Its separate local process keeps webcam frames in memory. Recorded eight-second webcam segments took about 2.2 seconds to process in the integration check; no physical camera was activated. See `AUTOAVSR.md` and `reports/autoavsr-integration.json`. This integration does not change any LipNet checkpoint or evaluation result.
