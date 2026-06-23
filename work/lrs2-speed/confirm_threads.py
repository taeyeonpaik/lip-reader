import sys,json,time,gc,statistics
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch
from training import lrs2_finetune as t
from training.data import atomic_json
rows=json.loads((t.RUN/'manifest.json').read_text());train=sorted([r for r in rows if r['split']=='train'],key=lambda r:(r['declared_frames'],r['id']))
samples=[train[int((i+.5)*len(train)/16)] for i in range(16)]
e=t.Experiment(rows);path=t.STORAGE/'last.pt';before_sha=t.sha(path);reports=[]
for threads in [4,1,4,1]:
 saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 e.model.load_state_dict(saved['model'],strict=True);e.optimizer.load_state_dict(saved['optimizer']);del saved
 torch.set_num_threads(threads);e.mode(True)
 times=[]
 for step in range(2):
  t.seed(123+100+step);start=time.monotonic();e.optimizer.zero_grad(set_to_none=True)
  for r in samples:
   loss=e.loss(r);assert torch.isfinite(loss);(loss/16).backward()
  norm=torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.,error_if_nonfinite=True)
  e.optimizer.step();torch.mps.synchronize();times.append(time.monotonic()-start)
 reports.append(dict(threads=threads,seconds_per_update=times));print(reports[-1],flush=True)
 e.optimizer.zero_grad(set_to_none=True);gc.collect()
assert t.sha(path)==before_sha
summary={str(n):statistics.median([s for r in reports if r['threads']==n for s in r['seconds_per_update']]) for n in [4,1]}
atomic_json(Path('work/lrs2-speed/threads-confirmation.json'),dict(reports=reports,median_seconds_per_update=summary,speedup_ratio=summary['4']/summary['1'],checkpoint_unchanged=True,all_updates_discarded=True))
print(summary,flush=True)
