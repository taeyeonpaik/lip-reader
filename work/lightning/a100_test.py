"""Short training-only FP32/BF16 comparison. No production weights updated."""
import contextlib,fcntl,json,statistics,time
import torch
from training import lrs2_cuda as t
OUT=t.WORKSPACE/'work/lightning'
lock=(OUT/'a100-test.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'A100' in torch.cuda.get_device_name(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rows=t.prepare();e=t.Experiment(rows)
base=torch.load(t.ROOT/'models/autoavsr.pth',map_location='cpu',weights_only=True)
train=sorted(e.splits['train'],key=lambda r:(r['declared_frames'],r['id']))
samples=[train[min(len(train)-1,int((i+.5)*len(train)/16))] for i in range(16)]
assert all(r['split']=='train' for r in samples)
def mixed_loss(row):
 with torch.autocast('cuda',dtype=torch.bfloat16):x=e.encode(t.pixels(row,True))
 ids=[e.units.get(p,e.units['<unk>']) for p in e.sp.encode(row['text'],out_type=str)]
 assert len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))<=x.shape[1]
 labels=torch.tensor([ids],dtype=torch.long,device=x.device);length=torch.tensor([x.shape[1]],device=x.device)
 with torch.autocast('cuda',enabled=False):ctc,_=e.model.ctc(x.float(),length,labels)
 inp,out=t.add_sos_eos(labels,e.model.sos,e.model.eos,e.model.ignore_id)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred,_=e.model.decoder(inp,t.target_mask(inp,e.model.ignore_id),x,None)
 with torch.autocast('cuda',enabled=False):loss=.1*ctc+.9*e.model.criterion(pred.float(),out)
 return loss
report=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,samples=[r['id'] for r in samples],clip_frames=[r['declared_frames'] for r in samples],effective_batch=16,all_updates_discarded=True,no_validation_or_test_clips_used=True,arms={})
for precision in ['fp32','bf16']:
 try:
  e.model.load_state_dict(base,strict=True)
  e.optimizer=torch.optim.AdamW(e.model.parameters(),lr=1e-5,betas=(.9,.98),weight_decay=.03,foreach=False)
  e.mode(True);measurements=[]
  for step in range(3):
   t.seed(123+step);e.optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();losses=[]
   for row in samples:
    loss=e.loss(row) if precision=='fp32' else mixed_loss(row)
    assert torch.isfinite(loss);losses.append(float(loss.detach()));(loss/16).backward()
   norm=torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.,error_if_nonfinite=True)
   e.optimizer.step();torch.cuda.synchronize()
   m=dict(step=step,seconds=time.monotonic()-start,loss=statistics.mean(losses),gradient_norm=float(norm),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
   measurements.append(m);print(json.dumps(dict(precision=precision,**m)),flush=True)
  speed=statistics.median(m['seconds']/16 for m in measurements[1:])
  report['arms'][precision]=dict(passed=True,measurements=measurements,seconds_per_clip=speed,training_only_epoch_hours=speed*len(train)/3600)
 except Exception as error:
  report['arms'][precision]=dict(passed=False,error=f'{type(error).__name__}: {error}')
  print(json.dumps(report['arms'][precision]),flush=True)
 t.atomic_json(OUT/'a100-test-result.json',report)
assert t.sha(t.ROOT/'models/autoavsr.pth')==t.BASE_SHA256
report.update(original_weights_unchanged=True,completed_unix=time.time(),validation_time_not_included=True,bf16_full_training_validation_pending=True)
t.atomic_json(OUT/'a100-test-result.json',report);print(json.dumps(report),flush=True)
