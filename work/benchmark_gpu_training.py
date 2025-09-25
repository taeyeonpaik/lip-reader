import sys,time,json
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'outputs/lipnet-tutorial'))
from tutorial.accelerated import AcceleratedLipNet,greedy_decode,loss_ctc
torch.set_num_threads(4)
ref=root/'work/gpu-training-reference'
data=np.load(ref/'reference.npz')
frames=torch.from_numpy(data['frames'][None].copy())
targets=torch.from_numpy(data['labels'][None].copy())
lengths=torch.tensor([targets.shape[1]])
model=AcceleratedLipNet();model.load_keras_arrays(ref/'published.npz');model.eval()
report={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'parity':[],'benchmarks':[]}
for device in ['cpu','mps']:
 model.to(device)
 with torch.no_grad():logits=model(frames.to(device));prob=logits.softmax(-1).cpu().numpy()
 error=float(np.max(np.abs(prob-data['probabilities'])))
 result={'device':device,'maximum_probability_error':error,'prediction':greedy_decode(logits)[0],
         'ctc_loss':loss_ctc(logits,targets,lengths).item(),'keras_ctc_loss':float(data['loss'].reshape(-1)[0])}
 assert error<.002,result
 assert result['prediction']=='bin red at s nine again',result
 report['parity'].append(result);print(json.dumps(result),flush=True)
model.to('cpu');del model
torch.mps.empty_cache()
for batch_size,amp in [(2,False),(2,True),(4,True)]:
 model=AcceleratedLipNet();model.load_keras_arrays(ref/'initial.npz');model.to('mps').train()
 optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-4,eps=1e-7)
 scaler=torch.amp.GradScaler('mps',enabled=amp,init_scale=128)
 inputs=frames.repeat(batch_size,1,1,1,1).to('mps')
 labels=targets.repeat(batch_size,1);label_lengths=lengths.repeat(batch_size)
 measurements=[]
 for step in range(4):
  start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
  with torch.autocast('mps',dtype=torch.float16,enabled=amp):logits=model(inputs)
  loss=loss_ctc(logits,labels,label_lengths)
  scaler.scale(loss).backward();scaler.unscale_(optimizer)
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
  scaler.step(optimizer);scaler.update();torch.mps.synchronize()
  result={'batch_size':batch_size,'amp':amp,'step':step,'seconds':time.perf_counter()-start,
          'loss':float(loss),'gradient_norm':float(norm),'scale':scaler.get_scale()}
  assert np.isfinite(result['loss']),result
  print(json.dumps(result),flush=True);measurements.append(result)
 report['benchmarks'].append(measurements)
 del inputs,optimizer,scaler,logits,loss,model
 torch.mps.empty_cache()
(root/'outputs/lipnet-tutorial/validation/gpu-training-parity.json').write_text(json.dumps(report,indent=2)+'\n')
