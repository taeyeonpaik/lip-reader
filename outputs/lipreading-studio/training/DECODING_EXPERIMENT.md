# Wider search, external language model, and expanded corpora

## Completed evaluation

All decoder comparisons and the frozen cross-dataset audit finished. Beam 20
without LM was selected on validation before the audits. The app retains beam 5
because the observed gains are tiny and the wider beam costs more processing.
The external-LM finalist regressed on validation CER, failing the selection rule.

| Evaluation set | Beam 5 WER | Beam 20 WER | Change in word errors |
| --- | ---: | ---: | ---: |
| LRS2 validation, 500 | 21.404% | 21.219% | −7 |
| LRS2 test, 1,243 | 16.186% | 16.141% | −3 |
| Extra LRS2 validation, 582 | 24.444% | 24.419% | −1 |
| LRS3 session validation, 500 | 8.915% | 8.897% | −1 |

Extra LRS2 CER worsened from 15.688% to 15.774%. All 14 reports were independently
rescored, including both cross-dataset comparisons, with zero recognition failures.
The LRS3 subset is part of trainval, already used by the base model's authors;
its sub-10% WER is not a new benchmark result or a personal-video accuracy claim.

Data expansion and independent verification are complete. Verified counts are
45,814 LRS2 and 30,468 LRS3 training clips, 2,570 validation clips and 1,243 LRS2
test clips. One LRS3 clip (`kiUnJ1d8vvw/50004`) missing from the independent
reference was excluded. Use `work/corpora/verified-manifest.json`. Corpus media
is not packaged. Mixed fine-tuning subsequently completed as experiment 4; see
[MIXED_EXPERIMENT.md](MIXED_EXPERIMENT.md).

## Decoder implementation

`studio/pipeline.py` supports beam widths 1–80, external LM weight, CTC weight,
and length bonus. Defaults remain exactly the prior beam-5 / CTC .1 / attention
.9 configuration. `studio/language_model.py` loads the authors' English 16-layer
Transformer LM with 53,742,649 parameters and 5,049 token IDs. Its vocabulary is
verified against the authors' token file and every checkpoint key loads strictly.
Files and hashes are under `models/english-lm/`.

The external LM scores candidate text during beam search (shallow fusion).
It does not rewrite a finished transcript and never receives audio. The visual
model weights are unchanged: this is a decoding improvement experiment, not a
new claim of fine-tuned visual weights.

Engineering checks cover causal masking, cached versus full-prefix scoring,
beam branching/reordering, exact agreement with the author's layer equations,
valid scores on real clips, and invalid configuration rejection. Twenty-two tests
passed; detailed reports are in `validation/decoding-*`.

## Fixed comparison

1. Use the first 100 clips of the existing deterministic 500-clip LRS2 validation
   set as a pilot. Compare beam 5/no LM; beam 20/no LM; beam 20/LM .15; beam 20/LM
   .30; beam 40/LM .30. Keep CTC .1 and length bonus zero throughout.
2. Evaluate the two pilot candidates with lowest WER on all 500 validation clips,
   alongside baseline. Select lower WER with CER no worse than baseline and zero
   failures. Tie-break by runtime. Fix selection before test evaluation.
3. Evaluate baseline and the selected decoder on all 1,243 LRS2 test clips. This
   test baseline was already examined in previous work; no clean-new-benchmark
   claim. Do not tune settings against test predictions.
4. After corpus acquisition, independently check LRS3 annotations, then audit the
   frozen selected decoder on the 582 additional LRS2 validation clips and up to
   500 session-held-out LRS3 clips. No new hyperparameter search on this audit.
5. Review errors, runtime and webcam segment throughput. Activate a measured
   improvement only after verification; keep the original option available if
   the slower configuration cannot keep up with webcam capture. A result that
   improves one corpus but regresses on another must be reported explicitly.

The pilot and full validation results are not independent. The extra LRS2 audit
clips were not part of the previous local adaptation's validation subset. LRS3
session holdout is only a local partition; the pretrained model already used
LRS3 and LRS2, and speaker identities are not independently verified.

## Dataset expansion

`training/expand_corpora.py` downloads all 22 LRS2 main-training parquet shards
from pinned public MahmoodAnaam releases, preserving the original validation and
test partitions. It also imports the separately public processed release
`TheNHz/ellipsis-lrs3-trainval-autoavsr` at revision
`a9b5374e19082b3217171be8ece589f35c8a2a71`.

LRS3: 31,957 available clips, approximately 30 hours, from trainval only. The
407-hour pretrain split and official LRS3 test split are **not included**. LRS3
crops are grayscale 96×96 at 25 fps, aligned by the distributor with Auto-AVSR
landmarks. Normalize after center-cropping to 88×88; do not apply RGB grayscale
weights a second time. `training/inputs.py` and `training/deeper.py` support both
LRS2 RGB MP4 and LRS3 grayscale NPZ for subsequent mixed-data training.

Five percent of LRS3 recording sessions are reserved for local validation by a
fixed hash of session ID. No predictions influence the split. Preserve namespaces
and check duplicate IDs, video hashes, and recording overlap. Compress imported
arrays locally and discard each temporary downloaded tar after verifying its
hash/size and import ledger, keeping free space above 12 GiB.

The processed LRS3 sample metadata contains generic fields such as `source:
voxpopuli` that conflict with the release README. Before use, run
`training.verify_corpora` against the independent Llama-AVSR authors' public
30-hour LRS3 CSV. All 31,956 retained clips passed ID, frame-count and transcript checks; the unmatched clip was excluded. Do not access gated raw mirrors to resolve this.

Dataset videos stay in workspace caches and must not be packaged with the app.
Subsequent mixed fine-tuning is recorded separately in experiment 4.

## Running commands and recovery

From the workspace root, with the project's `work/venv`:

```sh
PYTHONPATH=outputs/lipreading-studio caffeinate -i work/venv/bin/python -u \
  -m training.expand_corpora --destination work/corpora \
  --lrs2-downloads work/overnight/data --existing-videos work/deeper/dataset/videos

PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.decoding_experiment \
  --manifest work/overnight/dataset/manifest.json --cache work/overnight/features \
  --run outputs/lipreading-studio/training/runs/decoding-v3
```

These were launched as separate managed foreground processes, with logs at
`work/decoding/corpora.log` and `work/decoding/experiment.log`. Do not start a
second copy while either corresponding process is alive. The decoder has a run
lock. Data acquisition resumes from verified local downloads and shard ledgers.

When both finish:

```sh
PYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.verify_corpora \
  --manifest work/corpora/verified-manifest.json \
  --labels work/decoding/lrs3-author-30h-labels.csv \
  --destination outputs/lipreading-studio/validation/combined-data-label-audit.json

PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=work/matplotlib-cache \
  caffeinate -i work/venv/bin/python -u -m training.cross_dataset_audit \
  --manifest work/overnight/dataset/manifest.json \
  --combined-manifest work/corpora/verified-manifest.json --cache work/overnight/features \
  --run outputs/lipreading-studio/training/runs/decoding-v3
```

GPU/graphics and network commands need access outside the sandbox on this Mac.
After verification, copy `work/corpora/verified-provenance.json` to
`training/combined-data-provenance.json`, and include a split manifest with IDs
and hashes but no local paths or raw media. Keep original model weights. The app
reads an optional `models/decoding-selection.json` with `decoding` keyword values
and a human-readable `description`; create this only after the result supports
activation. Update review docs and rerun package script at
`work/overnight/package.py` after final checks. Pause the monitor when done.

Sources: [Author inference configuration](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/configs/LRS3_V_WER19.1.ini),
[public processed LRS3 release](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-trainval-autoavsr),
[independent LRS3 labels](https://github.com/umbertocappellazzo/Llama-AVSR#3-label-files-download).

## Final runtime checks

On the included eight-second video, two uploads including normalization, face
tracking and recognition took **4.47 / 3.03 seconds with beam 5**, versus
**4.72 / 4.67 seconds with beam 20**. Model loading is excluded.
Two paced eight-second webcam windows per decoder processed in **2.13 / 2.10
seconds with beam 5** and **3.82 / 4.41 seconds with beam 20**, with zero skipped
windows. This is a recorded-video throughput check, not a physical-camera test
or a guarantee for other speakers. Timing details are in
`validation/decoding-throughput.json`. The faster beam-5 default is retained.

