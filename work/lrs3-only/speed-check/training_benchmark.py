import sys,json,time,gc,hashlib,random
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'outputs/lipreading-studio'))
from training.lrs3_finetune import Experiment,RUN,seed,verify_cache
import torch
rows=json.loads((RUN/'manifest.json').read_text())
train=sorted((r for r in rows if r['split']=='train'),key=lambda r:(r['declared_frames'],r['id']))
cache=Path('/private/tmp/lrs3-io-benchmark');cache.mkdir(exist_ok=True)
samples=[]
for i in range(16):
 row=dict(train[int((i+.5)*len(train)/16)])
 verify_cache(row)
 raw=Path(row['path']).read_bytes();path=cache/Path(row['path']).name;path.write_bytes(raw);row['path']=str(path);samples.append(row)
exp=Experiment(rows);exp.mode(True);results=[]
for step in range(3):
 seed(123+step);exp.optimizer.zero_grad(set_to_none=True);torch.mps.synchronize();tick=time.perf_counter();loss_sum=0.
 for row in samples:
  loss=exp.loss(row);assert torch.isfinite(loss);loss_sum+=float(loss.detach());(loss/16).backward()
 norm=torch.nn.utils.clip_grad_norm_(exp.model.parameters(),1.,error_if_nonfinite=True)
 exp.optimizer.step();torch.mps.synchronize();seconds=time.perf_counter()-tick
 result=dict(step=step,clips=16,frames=sum(r['declared_frames'] for r in samples),seconds=seconds,seconds_per_clip=seconds/16,mean_loss=loss_sum/16,gradient_norm=float(norm),mps_driver_gib=torch.mps.driver_allocated_memory()/1024**3)
 results.append(result);print(json.dumps(result),flush=True)
report=dict(measurements=results,samples=[{'id':r['id'],'frames':r['declared_frames']} for r in samples],no_weights_saved=True,benchmark_updates_discarded=True,recipe='same full-model float32 loss, microbatch1 accumulation16, CPU decoder and MPS encoder',note='Length-stratified 16 training clips, local byte-identical copies; does not measure cold cloud I/O, sustained thermal slowdown, checkpoints or validation.')
(root/'work/lrs3-only/speed-check/training-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
