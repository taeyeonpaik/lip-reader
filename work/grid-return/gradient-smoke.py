import json
from pathlib import Path
import torch,numpy as np
from tutorial.runtime import GridReader,ROOT
from tutorial.accelerated import loss_ctc,VOCABULARY
r=GridReader();m=r.model;m.train();paths=list(Path('work/grid-return/cache').glob('*.npz'))[:2]
x=torch.from_numpy(np.stack([np.load(p)['frames'] for p in paths])).to(r.device)
labels=[[VOCABULARY.index(c)+1 for c in 'bin red at s nine again']]*2
with torch.autocast(r.device,dtype=torch.float16): y=m(x)
loss=loss_ctc(y,torch.tensor(sum(labels,[])),torch.tensor([len(l) for l in labels]))
scaler=torch.amp.GradScaler(r.device,init_scale=128);opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=1e-5,eps=1e-7)
scaler.scale(loss/4).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
assert torch.isfinite(loss) and torch.isfinite(norm)
scaler.step(opt);scaler.update()
report={'loss':float(loss.detach()),'gradient_norm_before_clipping':float(norm),'finite_gradients':True,'mps_optimizer_step_passed':True,'saved_weights_modified':False,'note':'Synthetic labels on two cached training inputs; numerical smoke only, not accuracy.'}
(ROOT/'validation/grid/gradient-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
