from pathlib import Path
import json
root=Path('outputs/lipreading-studio')
summary='''## VideoMAE transfer learning — completed September 18

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

'''
for name in ['README.md','REVIEW.md']:
 p=root/name;s=p.read_text();start=s.index('## VideoMAE transfer learning');end=s.index('\n## ',start+4) if '\n## ' in s[start+4:] else -1
 # Original inserted section ends before the old document content, not necessarily a header.
 marker='The distributable ZIP will be refreshed after final review; its current version\npredates VideoMAE.\n\n'
 end=s.index(marker,start)+len(marker)
 s=s[:start]+summary+s[end:]
 s=s.replace('a separate VideoMAE transfer-learning model is being trained.', 'a completed VideoMAE transfer-learning experiment is retained for review.')
 p.write_text(s)
p=root/'training/VIDEOMAE_EXPERIMENT.md';s=p.read_text();s=s.replace('This document will be updated with final results after the bounded run.', 'Training and evaluation are complete; final results and review follow below.')
local=summary.replace('(training/VIDEOMAE_EXPERIMENT.md)','(VIDEOMAE_EXPERIMENT.md)')
s=s.replace('## Existing paired training datasets',local+'## Existing paired training datasets',1)
s=s.replace('The final trained\nmodel selector path remains a post-training review step.', 'The final trained\nmodel selector, failure warning and example inference were reviewed in the app.')
s=s.replace('Final model\nruntime will be measured again.', 'Final model\nruntime is reported in the completed-results section above.')
p.write_text(s)
p=Path('outputs/START_HERE.md');s=p.read_text();a=s.index('The current experiment trains');b=s.index('The following results describe',a)
s=s[:a]+'''The VideoMAE experiment is **complete and unsuccessful**: 32 epochs over 5,000
clips yielded 117.18% WER on the 200-clip LRS2 test subset and 115.34% on the
200-clip local LRS3 audit. The existing Auto-AVSR model scored 16.41% and 8.62%
on those same clips and remains the default. LRS3 is not the official benchmark.
The VideoMAE option is retained as an explicitly failed research experiment.
No additional training is running. [Full VideoMAE results](lipreading-studio/training/VIDEOMAE_EXPERIMENT.md).

'''+s[b:]
s=s.replace('VideoMAE training is a separate new experiment.', 'The separate VideoMAE experiment is now also complete.')
p.write_text(s)
