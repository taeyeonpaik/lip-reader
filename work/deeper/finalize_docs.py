from pathlib import Path
import json,re
root=Path('outputs/lipreading-studio')
run=root/'training/runs/lrs2-deeper-v2'
r=json.loads((run/'result.json').read_text())
summary='''## Completed deeper fine-tuning — September 17

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

'''
review=(root/'REVIEW.md').read_text()
review=re.sub(r'> \*\*September 17 update:.*?\n\n','',review,count=1)
review=review.replace('## Completed training and evaluation',summary+'## Earlier output-layer experiment (experiment 1)',1)
review=review.replace('It is a\nsubstantial extension beyond the original LipNet tutorial, as agreed.','It uses a different architecture from the original LipNet tutorial, as agreed.')
review=review.replace('- Implemented reproducible output-layer fine-tuning of pretrained Auto-AVSR on\n  4,000 LRS2 clips; used held-out validation to retain the stronger baseline and\n  measured 16.2% word error on 1,243 prepared-mouth test clips.','- Implemented and evaluated two Auto-AVSR fine-tuning experiments, including\n  55.1 million trainable parameters on 12,000 LRS2 clips over 10 epochs; compared\n  validation and test errors and retained the pretrained baseline when test word\n  recognition did not improve.')
review=review.replace('## Evidence and source\n','## Evidence and source\n\n- [Deeper experiment results](training/runs/lrs2-deeper-v2/result.json)\n- [Deeper training history](training/runs/lrs2-deeper-v2/history.json)\n- [Checkpoint and metric verification](validation/deeper-checkpoint-and-metrics.json)\n- [Deeper training recipe](training/DEEPER_EXPERIMENT.md)\n')
(root/'REVIEW.md').write_text(review)
# Make the root report's relative project links resolve correctly.
outer=re.sub(r'\]\((?!https?://)([^)]+)\)',r'](lipreading-studio/\1)',review)
Path('outputs/MORNING_REVIEW.md').write_text(outer)
for name,link in [('README.md','training/DEEPER_EXPERIMENT.md'),('training/README.md','DEEPER_EXPERIMENT.md')]:
 p=root/name;s=p.read_text()
 s=re.sub(r'> \*\*September 17 update:.*?\n\n',f'> **Completed September 17:** Deeper fine-tuning finished 10 epochs and 15,000 updates. Validation WER improved from 21.40% to 21.06%, while test WER worsened from 16.19% to 16.25%. The app retains the pretrained model. [Full experiment results]({link}). The earlier output-layer experiment is also documented below.\n\n',s,count=1)
 p.write_text(s)
p=root/'training/DEEPER_EXPERIMENT.md';s=p.read_text()
start=s.index('## Status');end=s.index('## What changes')
s=s[:start]+summary+s[end:]
s=s.replace('## Resume locally','## Reproduce or resume locally')
s=s.replace('From the workspace root:\n\n```sh','From the workspace root, acquire the six pinned training shards and original validation/test files if needed:\n\n```sh\nPYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.download \\\n  --destination work/deeper/data --deeper\nPYTHONPATH=outputs/lipreading-studio work/venv/bin/python -m training.data \\\n  --downloads work/deeper/data/downloads.json --destination work/deeper/dataset \\\n  --train-limit 12000 --validation-limit 500\n```\n\nResume the existing run with its original manifest; choose a new run directory for a fresh experiment or a manifest reconstructed at a different absolute path:\n\n```sh')
s=s.replace('An hourly follow-up is configured to inspect progress, recover interruptions, and complete the final review. It should remain quiet when nothing actionable changes and pause when the experiment and deliverables are finished.','The experiment and final review are complete. No additional training is scheduled. The monitoring follow-up is being closed after final packaging.')
p.write_text(s)
p=Path('outputs/START_HERE.md');s=p.read_text();start=s.index('Local fine-tuning completed:');end=s.index('\nStart with',start)
s=s[:start]+'''Two local fine-tuning experiments completed. The deeper run trained 55.1 million
parameters on 12,000 LRS2 clips for 10 epochs and 15,000 updates. Validation word
error improved from 21.40% to 21.06%, but test word error worsened from 16.19% to
16.25%. The app retains the original pretrained model. Its measured test word
error is **16.19%** on prepared mouth crops. The goal of improving test word
recognition remains unmet.
'''+s[end:];p.write_text(s)
selection={'selected':'pretrained','reason':'Deeper fine-tuning improved validation WER by 0.34 points, below the preset 0.50-point threshold; candidate test WER was 16.25% versus 16.19% baseline','experiment':'training/runs/lrs2-deeper-v2','base_sha256':r['config']['base_sha256']}
(root/'models/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
