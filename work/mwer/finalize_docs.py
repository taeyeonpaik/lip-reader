from pathlib import Path
import json,re
root=Path('outputs/lipreading-studio')
summary='''## Final MWER pilot completed — September 18

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

**This is the final training attempt under the user's instruction.** No further
training campaign or deployment is planned. The app remains usable locally,
and all candidate checkpoints, split provenance, predictions and experiment
code are retained for review. Dataset media is excluded from the app archive.

'''
for name in ['README.md','REVIEW.md']:
 p=root/name;s=p.read_text();s='\n'.join(line for line in s.splitlines() if not line.startswith('> **Final bounded attempt running:'))+'\n'
 at=s.index('## Completed mixed-data')
 s=s[:at]+summary+s[at:];p.write_text(s)
p=Path('outputs/START_HERE.md');s=p.read_text();s='\n'.join(line for line in s.splitlines() if not line.startswith('> **Final bounded attempt running:'))+'\n'
s=s.replace('The final MWER pilot is described above; the site remains undeployed.', 'The final MWER pilot is complete: it corrected 11 validation word errors, while the matched conventional-training control corrected 14. Neither met the preset gain threshold. The app keeps pretrained weights, and no further training is running. The site remains undeployed.')
p.write_text(s)
p=root/'training/MWER_EXPERIMENT.md';s=p.read_text();a=s.index('Implementation and real-data');b=s.index('## Question and diagnostic');s=s[:a]+summary+'\n'+s[b:]
s=s.replace('Evidence is in\n`validation/mwer-feasibility.json` once the checks pass.', 'The real-data checks passed; evidence is in\n`validation/mwer-feasibility.json`. All 26 regression tests passed.')
a=s.index('## Finish');s=s[:a]+'''## Completion and retention

The pilot is complete. Both arms and all four epoch checkpoints are retained.
Neither qualified for the conditional audit/test step. Original weights and the
fast app decoder remain active; no format-3 adapter was activated. Independent
metric, checkpoint and uncertainty evidence is published in
`validation/mwer-checkpoint-and-metrics.json`. The final review and ZIP include
this experiment. No further training is authorized without a new user request.
''';p.write_text(s)
p=root/'models/selection.json';d=json.loads(p.read_text());d['experiment']='training/runs/mwer-v5';d['reason']='Final MWER pilot reduced mean validation WER by 0.289 points, below the preset 0.50-point threshold, and underperformed the matched conventional-training control (0.367-point gain). No candidate qualified; original pretrained weights and fast decoder retained.';p.write_text(json.dumps(d,indent=2)+'\n')
Path('outputs/MORNING_REVIEW.md').write_text(re.sub(r'\]\((?!https?://|#)([^)]+)\)',r'](lipreading-studio/\1)',(root/'REVIEW.md').read_text()))
