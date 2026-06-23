import sys,time,json,gc,statistics
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch
from training import lrs2_finetune as t
from training.data import atomic_json
out=Path('work/lrs2-speed');rows=json.loads((t.RUN/'manifest.json').read_text());train=sorted([r for r in rows if r['split']=='train'],key=lambda r:(r['declared_frames'],r['id']))
sample=[train[int((i+.5)*len(train)/16)] for i in range(16)]
e=t.Experiment(rows)
# Same samples and augmentation seeds in each execution variant. No optimizer
# update is retained, and validation/test are never used for benchmarking.
base_loss=e.loss

def gpu_loss(row,augment=True):
    x=e.model.proj_encoder(e.model.frontend(t.pixels(row,augment)))
    x,_=e.model.encoder(x,None)
    ids=[e.units.get(p,e.units['<unk>']) for p in e.sp.encode(row['text'],out_type=str)]
    labels=torch.tensor([ids],dtype=torch.long)
    ctc,_=e.model.ctc(x.cpu(),torch.tensor([x.shape[1]]),labels)
    labels=labels.to('mps')
    inp,target=t.add_sos_eos(labels,e.model.sos,e.model.eos,e.model.ignore_id)
    pred,_=e.model.decoder(inp,t.target_mask(inp,e.model.ignore_id),x,None)
    return .1*ctc+.9*e.model.criterion(pred,target).cpu()

reports=[]
for device,threads in [('cpu',4),('cpu',1),('cpu',2),('mps',1),('mps',2)]:
    print('variant',device,threads,flush=True)
    e.model.decoder.to(device);e.model.criterion.to(device);torch.set_num_threads(threads)
    lossfn=base_loss if device=='cpu' else gpu_loss
    times=[]
    try:
        e.mode(True)
        # One warm-up backward, discarded.
        e.optimizer.zero_grad(set_to_none=True);t.seed(8300)
        lossfn(sample[8]).backward();torch.mps.synchronize();e.optimizer.zero_grad(set_to_none=True)
        for repeat in range(2):
            tick=time.monotonic();losses=[]
            for i,row in enumerate(sample):
                t.seed(8400+i)
                loss=lossfn(row)
                (loss/16).backward();losses.append(float(loss.detach()))
            torch.mps.synchronize()
            elapsed=time.monotonic()-tick
            finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.model.parameters())
            assert finite
            times.append(elapsed);e.optimizer.zero_grad(set_to_none=True)
        report=dict(device=device,threads=threads,seconds_per_16=times,median_seconds_per_clip=statistics.median(times)/16,finite_gradients=True)
    except Exception as exc:
        report=dict(device=device,threads=threads,error=repr(exc));e.optimizer.zero_grad(set_to_none=True)
    reports.append(report);print(json.dumps(report),flush=True)
    atomic_json(out/'benchmark.json',dict(samples=[dict(id=r['id'],frames=r['declared_frames']) for r in sample],reports=reports,optimizer_updates=0,all_benchmark_gradients_discarded=True))
    gc.collect();torch.mps.empty_cache()
print('COMPLETE',flush=True)
