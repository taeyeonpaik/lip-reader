from pathlib import Path
import json,hashlib,shutil
root=Path('outputs/lipreading-studio')
run=root/'training/runs/lrs2-heads-v1'
r=json.loads((run/'result.json').read_text())
h=json.loads((run/'history.json').read_text())
assert r['state']=='completed' and r['epochs_completed']==len(h)
assert r['selection']['selected']=='pretrained'
assert r['baseline_test']['clips']==1243 and r['baseline_test']['failures']==0
selection={'selected':'pretrained','reason':r['selection']['reason'],'experiment':'training/runs/lrs2-heads-v1','base_sha256':r['config']['base_sha256']}
(root/'models/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
status={'state':'completed','real_fine_tuning_started':True,'real_fine_tuning_completed':True,'accuracy_improvement_established':False,'selected_model':'pretrained','epochs_completed':r['epochs_completed'],'optimizer_updates':r['training_steps'],'test_wer':r['baseline_test']['wer'],'test_cer':r['baseline_test']['cer'],'test_clips':1243,'run':'runs/lrs2-heads-v1','data':'data-provenance.json'}
(root/'training/status.json').write_text(json.dumps(status,indent=2)+'\n')
shutil.copyfile('work/overnight/tests.log',root/'validation/tests.txt')
manifest=json.loads(Path('work/overnight/dataset/manifest.json').read_text())
(root/'training/split-manifest.json').write_text(json.dumps([{k:v for k,v in row.items() if k not in ['path','text']} for row in manifest],indent=2)+'\n')
for file in ['README.md','training/README.md']:
 p=root/file;s=p.read_text()
 if file=='README.md':
  s=s.replace('See [training instructions](training/README.md), the live **Model & training** tab,\nand `training/runs/lrs2-heads-v1/result.json` once the experiment completes.', 'The completed run trained three full epochs and 1,500 optimizer updates. It did\nnot improve validation word error, so the app keeps the original pretrained\nmodel. That model measured **16.19% word error**, **10.24% character error** and\n**728/1,243 exact transcripts** on the prepared-mouth LRS2 test split, with no\nfailed examples. See [the review report](REVIEW.md), [training instructions](training/README.md)\nand `training/runs/lrs2-heads-v1/result.json`.')
  s=s.replace('plus processing\n time','plus processing\ntime')
 else:
  s=s.replace('## Experiment','## Completed result\n\nThree full epochs and 1,500 optimizer updates completed in about 39.5 minutes.\nValidation WER did not improve (21.40% baseline; 21.40%, 21.54%, 21.48% by epoch),\nso early stopping retained the pretrained model. The unchanged model achieved\n16.19% WER and 10.24% CER on 1,243 prepared-mouth test clips, with 728 exact\ntranscripts and no failures. Real fine-tuning completed; no accuracy improvement\nwas established.\n\n## Experiment',1)
 p.write_text(s)
p=Path('outputs/START_HERE.md');s=p.read_text().replace('Real local fine-tuning is running on 4,000 LRS2 training clips, with 500 validation\nclips and 1,243 separate test clips. The original pretrained weights are preserved.\nThe app remains usable while the experiment runs.', 'Local fine-tuning completed: three epochs on 4,000 LRS2 training clips, with 500\nvalidation and 1,243 separate test clips. It did not improve validation accuracy,\nso the app keeps the original pretrained model. Its measured test word error is\n**16.19%** on prepared mouth crops.\n\nStart with the [morning review report](lipreading-studio/REVIEW.md).')
p.write_text(s)
print(json.dumps(status,indent=2))
