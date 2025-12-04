from pathlib import Path
import sys,time,json,resource
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
from training.deeper import Deeper,parser
import torch
args=parser().parse_args(['--manifest','work/overnight/smoke-manifest.json','--cache','work/deeper/features','--run','work/deeper/batch-smoke'])
e=Deeper(args);e.mode(True)
results=[]
for i in range(3):
 tick=time.monotonic();e.optimizer.zero_grad(set_to_none=True)
 rows=e.splits['train'][i:i+4];loss=e.loss_batch(rows);assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.visual+e.language)
 torch.nn.utils.clip_grad_norm_(e.visual+e.language,1.,error_if_nonfinite=True);e.optimizer.step()
 results.append({'clips':len(rows),'frames':[r['declared_frames'] for r in rows],'loss':float(loss.detach()),'seconds':time.monotonic()-tick})
report={'microbatch_checks':results,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'includes_frozen_feature_extraction':True}
Path('outputs/lipreading-studio/validation/deeper-batch-feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
