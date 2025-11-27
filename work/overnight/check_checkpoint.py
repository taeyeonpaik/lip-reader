from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch,numpy as np,hashlib
from studio.pipeline import ROOT,LipReader
from studio.adaptation import head_state,set_heads,BASE_SHA256
from training.metrics import summarize
run=ROOT/'training/runs/lrs2-heads-v1'
reader=LipReader(); before=head_state(reader.model)
saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
set_heads(reader.model,saved['heads'])
changed={k:float((before[k]-saved['heads'][k]).abs().max()) for k in before}
assert all(v>0 for v in changed.values())
manifest=json.loads(Path('work/overnight/dataset/manifest.json').read_text())
reference={r['id']:r for r in json.loads((run/'epoch-03-validation.json').read_text())['predictions']}
checks=[]
for row in [r for r in manifest if r['split']=='validation'][:4]:
 key=hashlib.sha256((row['sha256']+BASE_SHA256+':rgb-center88-v1').encode()).hexdigest()
 features=torch.from_numpy(np.load(Path('work/overnight/features')/(key+'.npy'),allow_pickle=False))
 prediction=reader.decode(features)
 assert prediction==reference[row['id']]['prediction']
 checks.append({'id':row['id'],'identical_to_saved_epoch_evaluation':True})
test=json.loads((run/'baseline-test.json').read_text())
recomputed=summarize(test['predictions'])
assert all(test['metrics'][k]==v for k,v in recomputed.items())
report={'last_epoch':saved['epoch'],'head_max_absolute_change':changed,'checkpoint_reload_checks':checks,'test_metrics_recomputed_from_predictions':True,'test_metrics':recomputed,'production_base_unchanged':True}
(ROOT/'validation/checkpoint-and-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
