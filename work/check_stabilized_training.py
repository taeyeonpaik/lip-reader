import sys,json,time,random
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipnet').resolve()))
import torch
from lipnet.model import LipNet,ModelConfig
from lipnet.data import collate_batch
from lipnet.text import greedy_decode,ErrorRates
from full_train import CachedGrid,loss_for_batch
random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
device=torch.device('mps');torch.mps.manual_seed(42)
p=Path('work/paper-training-cache');d=json.loads((p/'manifest.json').read_text())
ds=CachedGrid(d['train'][:4],p,d['mean'],d['std']);batch=collate_batch([ds[i] for i in range(4)])
x=batch['videos'].to(device)
m=LipNet(ModelConfig(feature_norm=True,dropout=.25)).to(device)
opt=torch.optim.Adam(m.parameters(),lr=.0003)
scaler=torch.amp.GradScaler('mps',init_scale=128)
started=time.monotonic();history=[]
for step in range(1,601):
 m.train();opt.zero_grad(set_to_none=True)
 with torch.autocast('mps',dtype=torch.float16):out=m(x,batch['lengths'])
 loss=loss_for_batch(out,batch)
 scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
 scaler.step(opt);scaler.update()
 if step==1 or step%25==0:
  m.eval()
  with torch.no_grad(),torch.autocast('mps',dtype=torch.float16):out=m(x,batch['lengths'])
  preds=[greedy_decode(out[:batch['lengths'][i],i].float().cpu()) for i in range(4)]
  errors=ErrorRates()
  for ref,pred in zip(batch['texts'],preds):errors.update(ref,pred)
  record={'step':step,'loss':loss.item(),**errors.compute(),'predictions':preds,'seconds':time.monotonic()-started}
  history.append(record);print(json.dumps(record),flush=True)
  if record['cer']<.05 and len(set(preds))==4:break
result={'purpose':'Four-clip training-only overfit diagnostic; not a validation accuracy result','references':batch['texts'],'history':history,'passed':history[-1]['cer']<.05 and len(set(history[-1]['predictions']))==4}
Path('work/stabilized-overfit-result.json').write_text(json.dumps(result,indent=2))
