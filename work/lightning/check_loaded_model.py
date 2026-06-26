import json,time,hashlib
from pathlib import Path
import torch
from studio.pipeline import LipReader,ROOT
from training.inputs import load_pixels
root=ROOT.parents[1]
rows=json.loads((root/'work/corpora/lrs2/manifest.json').read_text())
r=next(r for r in rows if r['split']=='train' and (root/'work/corpora/lrs2/videos'/Path(r['path']).name).exists())
r['path']=str(root/'work/corpora/lrs2/videos'/Path(r['path']).name)
reader=LipReader(device='cpu',beam_size=40)
assert sum(p.numel() for p in reader.model.parameters())==250383410
with torch.inference_mode():
 x=load_pixels(r,'cpu');x=reader.model.proj_encoder(reader.model.frontend(x));x,_=reader.model.encoder(x,None)
 assert torch.isfinite(x).all()
 prediction=reader.decode(x.squeeze(0))
print(json.dumps(dict(passed=True,train_clip=r['id'],frames=r['declared_frames'],prediction=prediction,torch=torch.__version__,cuda_available=torch.cuda.is_available(),gradient_updates=0)),flush=True)
