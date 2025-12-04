from pathlib import Path
import sys,json,time,resource,hashlib
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
from training.deeper import Deeper,parser,selected_state
import torch,numpy as np
args=parser().parse_args(['--manifest','work/overnight/smoke-manifest.json','--cache','work/deeper/features','--run','work/deeper/smoke'])
e=Deeper(args); row=e.splits['train'][0];e.mode(False)
x=e.prefix(row)
with torch.no_grad(): encoded=e.encode_suffix(x).squeeze(0)
key=hashlib.sha256((row['sha256']+e.config['base_sha256']+':rgb-center88-v1').encode()).hexdigest()
reference=torch.from_numpy(np.load(Path('work/overnight/features')/(key+'.npy'),allow_pickle=False))
error=float((encoded-reference).abs().max());torch.testing.assert_close(encoded,reference,atol=2e-5,rtol=2e-5)
model=e.model
frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in ['encoder.encoders.0.norm_ff.weight','frontend.trunk.layer1.0.conv1.weight','decoder.decoders.0.norm1.weight']}
before=selected_state(model)
times=[]
e.mode(True)
for row in e.splits['train'][:4]:
 start=time.monotonic();e.optimizer.zero_grad(set_to_none=True);loss=e.loss(row);assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.visual+e.language)
 torch.nn.utils.clip_grad_norm_(e.visual+e.language,1.,error_if_nonfinite=True);e.optimizer.step()
 times.append({'loss':float(loss.detach()),'seconds':time.monotonic()-start})
for n,p in model.named_parameters():
 if n in frozen:torch.testing.assert_close(p.cpu(),frozen[n],atol=0,rtol=0)
after=selected_state(model)
changes={prefix:max(float((before[n]-after[n]).abs().max()) for n in before if n.startswith(prefix)) for prefix in e.config['prefixes']}
assert all(v>0 for v in changes.values())
report={'purpose':'Gradient, cache parity, and device feasibility; not measured accuracy improvement','trainable_parameters':e.config['trainable_parameters'],'prefix_reconstruction_max_abs_error':error,'updates':times,'updated_module_max_abs_changes':changes,'frozen_parameter_spot_checks':list(frozen),'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3}
Path('outputs/lipreading-studio/validation/deeper-feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
