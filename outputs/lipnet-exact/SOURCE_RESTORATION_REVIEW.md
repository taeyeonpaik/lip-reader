# Final source-restoration review — 19 September 2026

The two requested implementations now run separately using their original Keras source and published weights. The earlier custom alignment, PyTorch training adapter and grammar decoder are not active. Compatibility changes are itemized in `COMPATIBILITY.md`.

## Measured results

| Model and local sample | Word errors / reference words | WER | Exact sentences |
|---|---:|---:|---:|
| Notebook checkpoint, 100 s1 clips | 17 / 600 | **2.83%** | 83 / 100 |
| Paper unseen checkpoint, 50 s1 + 50 s2 clips, original beam + spelling | 111 / 600 | **18.50%** | 25 / 100 |
| Paper pipeline, above sample plus two upstream demos | 113 / 612 | 18.46% | 25 / 102 |
| Same 102 paper clips, raw beam output before original spelling correction | 190 / 612 | 31.05% | 10 / 102 |

No clip failed preprocessing or inference. The notebook's original demonstration `bras9a` predicts `bin red at s nine again`. The app's `bbaf2n` notebook example predicts `bin blue at f two now` exactly. The paper's supplied full-face demo predicts `set white with b two soon` against reference `set white with p two soon`; this error is preserved in the report.

The samples were chosen by a fixed hash rule, without optimizing model settings against their results. References are derived separately from GRID clip identifiers; they never enter inference. All predictions and an independently recomputed word-edit-distance audit are included.

### What these results establish

The notebook is a functioning low-WER demonstration on this s1 sample with the published checkpoint. Its complete training split is unknown, so **2.83% is not evidence of new-speaker performance**. The paper repository declares s1/s2/s20/s22 held out for its unseen-speaker scenario. Our 100 clips use only s1/s2 from that declared split. This is a small local subset, not the full published benchmark or a physical-webcam evaluation.

Conditional clip-bootstrap 95% intervals are 1.67–4.17% for the notebook and 15.83–21.33% for the paper subset. These intervals do not account for the wider population of speakers. The one/two-speaker samples are insufficient to establish general webcam accuracy. The paper subset has 18.00% WER on s1 and 19.00% on s2.

The previous 81.25% result was from a different custom pipeline on a different s4 test set. It is historical. These numbers must not be described as a controlled 81.25→2.83% improvement on the same benchmark. The current original models do not demonstrate sub-10% WER on new speakers.

## Implementation verification

- Pinned and verified 140 upstream source/asset hashes.
- Notebook model, loader and loss bodies executed directly from its source cells; separately instantiated model gives identical probabilities.
- Paper model2 loaded every original HDF5 tensor exactly; historical GRU defaults and CTC `1e-8` constant preserved.
- Independent GRU calculation maximum error: 2.98e-8. Historical RGB resize and CTC comparisons match exactly.
- Saving/reloading both models gives zero probability difference on checked inputs.
- Both original CTC losses produced finite gradients and changed weights in disposable optimizer checks. Published weight files are unchanged.
- App's upload path and recorded one-take worker give identical probabilities on the same RGB video. Practice prompts are kept out of inference.

## Runtime and product

The app is local at port 8502, with original examples, upload, webcam practice, the GRID sentence guide, random prompts and results reports. It uses CPU execution on this Mac. A 2.98-second paper-model upload took 4.18 seconds including mouth detection and decoding; its recorded-webcam equivalent took 4.23 seconds. Other local verification was running, so these are observed timings rather than isolated performance benchmarks.

Webcam mode records a three-second take and then processes it. It is not continuous real-time transcription. Physical-camera permissions, capture and accuracy have not been tested. Backend upload and recorded-webcam checks passed; browser interaction coverage is recorded separately in `reports/ui-checks.json`.

## Training status and project claims

**This stage restores and verifies the published implementations; it does not produce a newly trained checkpoint.** Explicit training entry points and unchanged original training sources are provided. The full source training procedures have not been rerun here. The disposable gradient checks are not training-result claims.

An accurate résumé description is: “Reproduced two LipNet implementations with verified legacy Keras compatibility; built a local GRID video-upload and webcam-practice app; audited published checkpoints on reproducible samples.” Claim 2.83% only with its s1/published-checkpoint qualification. Do not claim a newly invented model, own 96/178-epoch training, or general sub-10% WER.

Nothing has been deployed. Previous experiments and their negative findings remain archived separately.
