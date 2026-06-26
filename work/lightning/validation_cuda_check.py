"""Bounded, prediction-preserving validation execution comparison."""
import sys,json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'outputs/lipreading-studio'))
import torch
from studio.pipeline import LipReader
from training.inputs import load_pixels
p=ROOT/'outputs/lipreading-studio/training/runs/lrs2-batched-v1'
rows={r['id']:r for r in json.loads((p/'manifest.json').read_text())}
records=[json.loads(x) for x in (p/'baseline-validation.jsonl').read_text().splitlines() if x.strip()]
records=sorted(records,key=lambda r:rows[r['id']]['declared_frames'])
samples=[records[int((i+.5)*len(records)/12)] for i in range(12)]
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
reader=LipReader(device='cuda',beam_size=40);model=reader.model
features=[]
with torch.inference_mode():
 for record in samples:
  x=load_pixels(rows[record['id']],'cuda');x=model.proj_encoder(model.frontend(x));x,_=model.encoder(x,None);features.append(x[0])
results={}
for device,threads in [('cuda',1)]:
 torch.set_num_threads(threads)
 model.decoder.to(device);model.ctc.to(device);reader.search.to(device)
 measurements=[]
 with torch.inference_mode():
  for record,features_one in zip(samples,features):
   x=features_one.to(device);torch.cuda.synchronize();start=time.monotonic()
   hypotheses=reader.search(x);torch.cuda.synchronize()
   elapsed=time.monotonic()-start
   ids=hypotheses[0].yseq.tolist()[1:] if hypotheses else []
   prediction=''.join(reader.tokens[i] for i in ids if 0<i<len(reader.tokens)-1).replace('▁',' ').strip()
   measurements.append(dict(id=record['id'],frames=rows[record['id']]['declared_frames'],seconds=elapsed,original_seconds=record['seconds'],matches=prediction==record['prediction'],prediction=prediction,original_prediction=record['prediction']))
   print(json.dumps(dict(device=device,threads=threads,**measurements[-1])),flush=True)
 results[device]=measurements
(ROOT/'work/lightning/validation-cuda-check.json').write_text(json.dumps(results,indent=2)+'\n')
