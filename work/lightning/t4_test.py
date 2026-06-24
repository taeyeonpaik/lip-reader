"""Bounded training-only compatibility/throughput check; discard all updates."""
import gc,json,time,statistics,fcntl
from pathlib import Path
import torch
from training import lrs2_cuda as t
from studio.pipeline import LipReader
OUT=t.WORKSPACE/'work/lightning'
lock=(OUT/'t4-test.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert torch.cuda.is_available()
assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rows=t.prepare();e=t.Experiment(rows);e.mode(True)
train=sorted(e.splits['train'],key=lambda r:(r['declared_frames'],r['id']))
samples=[train[min(len(train)-1,int((i+.5)*len(train)/16))] for i in range(16)]
assert all(r['split']=='train' for r in samples)
results=[]
before={n:p.detach().cpu().clone() for n,p in e.model.named_parameters() if n in ('frontend.frontend3D.0.weight','proj_encoder.weight','encoder.encoders.0.feed_forward.w_1.weight','decoder.output_layer.weight','ctc.ctc_lo.weight')}
# One extreme-length probe verifies memory/gradient feasibility. It is discarded.
for probe in [True,False,False,False,False]:
 batch=[train[-1]] if probe else samples
 e.optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
 losses=[]
 for row in batch:
  loss=e.loss(row);assert torch.isfinite(loss);losses.append(float(loss.detach()));(loss/len(batch)).backward()
 norm=torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.,error_if_nonfinite=True)
 e.optimizer.step();torch.cuda.synchronize()
 result=dict(longest_probe=probe,clips=len(batch),seconds=time.monotonic()-start,frames=sum(r['declared_frames'] for r in batch),loss=statistics.mean(losses),gradient_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
 results.append(result);print(json.dumps(result),flush=True)
changed={n:not torch.equal(v,dict(e.model.named_parameters())[n].detach().cpu()) for n,v in before.items()};assert all(changed.values())
# Strict checkpoint/reload and optimizer-state recovery without altering production files.
e.optimizer.zero_grad(set_to_none=True)
checkpoint=OUT/'t4-test-only.pt'
t.save_torch(dict(model=e.model.state_dict(),optimizer=e.optimizer.state_dict()),checkpoint)
size=checkpoint.stat().st_size
probe=samples[8];e.mode(False);prediction=e.predict(probe)
del e,before;gc.collect();torch.cuda.empty_cache()
e=t.Experiment(rows);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
e.model.load_state_dict(saved['model'],strict=True);e.optimizer.load_state_dict(saved['optimizer']);del saved
e.mode(False);reloaded=e.predict(probe);assert reloaded==prediction
e.mode(True)
for parameter,state in e.optimizer.state.items():
 for key,value in state.items():
  if torch.is_tensor(value) and key!='step':assert value.device==parameter.device
# Confirm a reloaded accumulated update actually works; all 16 samples are training.
e.optimizer.zero_grad(set_to_none=True)
for row in samples:
 loss=e.loss(row);assert torch.isfinite(loss);(loss/16).backward()
torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.,error_if_nonfinite=True);e.optimizer.step();torch.cuda.synchronize()
assert t.sha(t.ROOT/'models/autoavsr.pth')==t.BASE_SHA256
seconds_per_clip=statistics.median(r['seconds']/r['clips'] for r in results[2:])
report=dict(passed=True,device=torch.cuda.get_device_name(0),torch=torch.__version__,measurements=results,seconds_per_clip=seconds_per_clip,training_only_epoch_hours=seconds_per_clip*len(train)/3600,validation_time_not_included=True,parameters=e.config['trainable_parameters'],representative_weights_changed=changed,strict_reload_prediction_match=True,optimizer_resume_update_passed=True,checkpoint_bytes=size,all_updates_discarded=True,no_validation_or_test_clips_used=True,original_weights_unchanged=True,completed_unix=time.time())
t.atomic_json(OUT/'t4-test-result.json',report);print(json.dumps(report),flush=True)
checkpoint.unlink()
