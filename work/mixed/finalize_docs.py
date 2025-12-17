from pathlib import Path
import json
root=Path('outputs/lipreading-studio')
summary='''## Completed mixed-data fine-tuning — September 18

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

'''
for name in ['README.md','REVIEW.md']:
 p=root/name; text=p.read_text(); lines=text.splitlines()
 lines=[line for line in lines if not line.startswith('> **Mixed-data fine-tuning is running')]
 text='\n'.join(lines)+'\n'
 text=text.replace('Experiment 4 is now running on the verified combined corpus, as described above.', 'Experiment 4 also completed; its results are described above.')
 if name=='README.md':
  pos=text.index('## Training and accuracy')
  text=text[:pos]+summary+text[pos:]
  text=text.replace('## Training and accuracy','## Earlier output-layer training and accuracy',1)
 else:
  pos=text.index('## Completed deeper fine-tuning')
  text=text[:pos]+summary+text[pos:]
 p.write_text(text)
# Keep a duplicate review at the output root, with relative links adjusted.
review=(root/'REVIEW.md').read_text()
import re
review=re.sub(r'\]\((?!https?://|#)([^)]+)\)',r'](lipreading-studio/\1)',review)
Path('outputs/MORNING_REVIEW.md').write_text(review)
p=root/'training/MIXED_EXPERIMENT.md'; text=p.read_text(); start=text.index('Launched September'); end=text.index('## Evidence motivating')
text=text[:start]+summary.split('## Completed decoding')[0]+'\n'+text[end:]
start=text.index('## Remaining completion work')
text=text[:start]+'''## Final review

Checkpoint and independent metric verification passed; see
`validation/mixed-checkpoint-and-metrics.json`. The app retains original weights.
The trained final checkpoint remains available for reproducibility, with no
adapted test result because none qualified on validation. Review documents and
the archive include the completed experiment and its negative overall outcome.
No further fine-tuning or deployment was started.
'''
p.write_text(text)
p=root/'training/DECODING_EXPERIMENT.md'; text=p.read_text(); start=text.index('## Current work'); end=text.index('## Decoder implementation')
text=text[:start]+'''## Completed evaluation

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

''' + text[end:]
text=text.replace('No additional visual fine-tuning run has been launched in this stage.', 'Subsequent mixed fine-tuning is recorded separately in experiment 4.')
text=text.replace("The first inspected clip's ID, transcript and frame count match.\nAll imported clips must be checked; report discrepancies rather than silently\nclaiming a verified corpus.", 'All 31,956 retained clips passed ID, frame-count and transcript checks; the unmatched clip was excluded.')
p.write_text(text)
p=Path('outputs/START_HERE.md'); text=p.read_text(); start=text.index('Two local fine-tuning'); end=text.index('Start with the')
text=text[:start]+'''Three local fine-tuning runs and a decoding comparison are complete. The latest
run used 76,282 LRS2/LRS3 clips for three epochs and 14,304 updates. LRS3 validation
WER improved from 8.74% to 8.38%, but LRS2 worsened slightly (22.88% → 22.92%).
No epoch qualified under the preset selection requirements. The app retains
pretrained weights and beam 5, measuring **16.19% LRS2 test WER** on prepared mouth
crops. A robust overall improvement over the pretrained model remains unmet.
No additional training is running; the site remains undeployed.

'''+text[end:];p.write_text(text)
