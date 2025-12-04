from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch
from training.deeper import Deeper,parser,restore,PREFIXES
from training.metrics import summarize
run=Path('outputs/lipreading-studio/training/runs/lrs2-deeper-v2')
result=json.loads((run/'result.json').read_text())
checked=[]
for p in sorted(run.glob('*.json')):
 d=json.loads(p.read_text())
 if isinstance(d,dict) and 'predictions' in d:
  assert summarize(d['predictions'])==d['metrics'],p
  assert d['metrics']['failures']==0,p
  checked.append(p.name)
args=parser().parse_args(['--manifest','work/deeper/dataset/manifest.json','--cache','work/deeper/features','--run',str(run)])
e=Deeper(args)
bundle=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
assert bundle['format']==2 and bundle['base_sha256']==e.config['base_sha256']
assert bundle['epoch']==result['selection']['best_epoch']==7
changes={prefix:max(float((t-e.original[n]).abs().max()) for n,t in bundle['state'].items() if n.startswith(prefix)) for prefix in PREFIXES}
assert all(v>0 for v in changes.values())
restore(e.model,bundle['state']);e.mode(False)
predictions=json.loads((run/'epoch-07-validation.json').read_text())['predictions']
lookup={r['id']:r for r in e.splits['validation']}
checks=[]
for expected in predictions[:4]:
 with torch.no_grad():x=e.encode_suffix(e.prefix(lookup[expected['id']])).squeeze(0)
 actual=e.reader.decode(x)
 assert actual==expected['prediction'],(actual,expected)
 checks.append({'id':expected['id'],'matches_recorded_prediction':True})
last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
assert last['epoch']==10 and last['step']==15000 and len(last['history'])==10
assert all(torch.isfinite(t).all() for t in last['state'].values())
assert last['history']==json.loads((run/'history.json').read_text())
hashes={}
for name in ['best.pt','last.pt']:
 with (run/name).open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
report={'state':'passed','metrics_recomputed':checked,'best_checkpoint_epoch':7,'last_checkpoint_epoch':10,'optimizer_updates':15000,'changed_modules_max_absolute_delta':changes,'reloaded_prediction_checks':checks,'checkpoint_sha256':hashes,'base_checkpoint_hash_verified':True,'production_model':'original pretrained'}
Path('outputs/lipreading-studio/validation/deeper-checkpoint-and-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
