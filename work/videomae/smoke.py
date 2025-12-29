import sys,time,json
from pathlib import Path
import numpy as np,torch,cv2
from studio.videomae import VideoFeatures,make_encoder,CTCHead,greedy,labels
from training.data import atomic_json

torch.set_num_threads(4)
state=torch.load('outputs/lipreading-studio/models/videomae-small-kinetics.pt',map_location='cpu',weights_only=True)
a=make_encoder(False).eval();a.load_state_dict(state)
b=make_encoder(True).eval();b.load_state_dict(state)
torch.manual_seed(66);x=torch.randn(1,3,16,224,224);mask=torch.zeros(1,1568,dtype=torch.bool)
with torch.inference_mode():
 y=a(x,mask);z=b(x,mask)
 delta=float((y-z).abs().max());assert torch.allclose(y,z,atol=1e-4,rtol=1e-4),delta
print('parity',delta,flush=True)
del a,b,x,y,z
rows=json.loads(Path('work/corpora/verified-manifest.json').read_text());row=next(r for r in rows if r['split']=='train' and 60<=r['declared_frames']<=80)
cap=cv2.VideoCapture(row['path']);frames=[]
while True:
 ok,f=cap.read()
 if not ok:break
 frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
cap.release();frames=np.stack(frames)
extractor=VideoFeatures();bench=[]
for i in range(2):
 t=time.perf_counter();f=extractor(frames);bench.append(time.perf_counter()-t);print('encode',i,len(frames),f.shape,bench[-1],flush=True)
head=CTCHead(); lens=torch.tensor([len(frames)]);tar=torch.tensor(labels(row['text']));opt=torch.optim.AdamW(head.parameters(),lr=1e-3)
losses=[]
for i in range(25):
 opt.zero_grad();logits=head(f.float().unsqueeze(0).clone(),lens);loss=torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),tar,lens,torch.tensor([len(tar)]));assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();losses.append(float(loss.detach()))
print('loss',losses[0],losses[-1],flush=True)
assert losses[-1]<losses[0]
report={'state':'passed','attention_parity_max_abs':delta,'device':extractor.device,'frames':len(frames),'encoding_seconds':bench,'one_clip_loss_before':losses[0],'one_clip_loss_after':losses[-1],'feature_shape':list(f.shape),'trainable_parameters':sum(p.numel() for p in head.parameters()),'reference':row['text']}
atomic_json('outputs/lipreading-studio/validation/videomae-smoke.json',report);print(json.dumps(report,indent=2),flush=True)
