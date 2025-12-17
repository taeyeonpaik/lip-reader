import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch
from training.deeper import Deeper,parser,selected_state,PREFIXES
from training.data import atomic_json
rows=json.loads(Path('work/corpora/verified-manifest.json').read_text())
mini=[]
for dataset in ['LRS2','LRS3-TED']:
 for split,limit in [('train',2),('validation',1),('test',1)]:mini.extend([r for r in rows if r['dataset']==dataset and r['split']==split][:limit])
atomic_json('work/mixed/smoke-manifest.json',mini)
a=parser().parse_args(['--manifest','work/mixed/smoke-manifest.json','--cache','work/deeper/features','--run','work/mixed/smoke','--balanced-datasets','--beam-size','20','--lr','3e-6','--accumulation','16','--epochs','6','--min-epochs','3','--patience','2','--mask-probability','.15','--seed','44'])
e=Deeper(a);e.mode(False)
for r in e.splits['train']:e.prefix(r)
before=selected_state(e.model);frozen=e.model.encoder.encoders[0].norm_ff.weight.detach().clone()
e.mode(True);e.optimizer.zero_grad(set_to_none=True);t=time.monotonic();loss=e.loss_batch(e.splits['train']);assert torch.isfinite(loss);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.visual+e.language)
torch.nn.utils.clip_grad_norm_(e.visual+e.language,1.,error_if_nonfinite=True);e.optimizer.step()
after=selected_state(e.model)
changes={p:max(float((before[n]-after[n]).abs().max()) for n in before if n.startswith(p)) for p in PREFIXES}
assert all(v>0 for v in changes.values());torch.testing.assert_close(e.model.encoder.encoders[0].norm_ff.weight,frozen,atol=0,rtol=0)
e.mode(False);metrics=e.evaluate('validation','smoke-validation');assert set(metrics['by_dataset'])=={'LRS2','LRS3-TED'};assert metrics['failures']==0
report={'state':'passed','purpose':'Mixed RGB and grayscale real-label gradient check, not an accuracy benchmark','loss':float(loss.detach()),'training_seconds':time.monotonic()-t,'batch_datasets':[r['dataset'] for r in e.splits['train']],'all_targeted_gradients_finite':True,'changed_modules':changes,'frozen_parameter_unchanged':True,'validation_aggregation_checked':True,'trainable_parameters':sum(p.numel() for p in e.visual+e.language)}
atomic_json('outputs/lipreading-studio/validation/mixed-training-feasibility.json',report);print(json.dumps(report,indent=2))
