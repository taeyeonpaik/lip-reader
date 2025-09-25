"""Bounded synthetic-label training feasibility test; never exports adapted weights."""
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import cv2,torch,sentencepiece
from studio.pipeline import LipReader,track_mouth,VENDOR
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
started=time.monotonic();reader=LipReader();model=reader.model
for p in model.parameters():p.requires_grad_(False)
for module in (model.ctc.ctc_lo,model.decoder.output_layer):
 for p in module.parameters():p.requires_grad_(True)
cap=cv2.VideoCapture('work/non-grid-silent.mp4');frames=[]
for _ in range(100):
 ok,f=cap.read()
 if not ok:break
 frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
cap.release()
import numpy as np
crops,_=track_mouth(np.stack(frames));pixels=torch.from_numpy(crops[:,4:92,4:92].copy()).float()/255
pixels=(pixels[...,0]*.2989+pixels[...,1]*.587+pixels[...,2]*.114)
pixels=((pixels-.421)/.165).unsqueeze(0).unsqueeze(2).to(reader.device)
with torch.no_grad():
 features=model.proj_encoder(model.frontend(pixels));features,_=model.encoder(features,None);features=features.cpu()
sp=sentencepiece.SentencePieceProcessor(model_file=str(VENDOR/'spm/unigram/unigram5000.model'))
units={s.split()[0]:int(s.split()[-1]) for s in (VENDOR/'spm/unigram/unigram5000_units.txt').read_text().splitlines()}
# Arbitrary labels exercise loss/backpropagation. They are not a transcript or evaluation reference.
labels=torch.tensor([[units.get(t,units['<unk>']) for t in sp.encode('THIS IS A TEST',out_type=str)]])
in_labels,out_labels=add_sos_eos(labels,model.sos,model.eos,model.ignore_id)
lengths=torch.tensor([features.shape[1]]);mask=torch.ones(1,1,features.shape[1],dtype=torch.bool)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
steps=[]
for step in range(3):
 tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
 ctc,_=model.ctc(features,lengths,labels)
 pred,_=model.decoder(in_labels,target_mask(in_labels,model.ignore_id),features,mask)
 att=model.criterion(pred,out_labels);loss=.1*ctc+.9*att
 assert torch.isfinite(loss);loss.backward()
 gradients=[p.grad for p in model.parameters() if p.requires_grad]
 assert all(g is not None and torch.isfinite(g).all() for g in gradients)
 torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
 optimizer.step()
 steps.append({'step':step+1,'seconds':time.monotonic()-tick,'synthetic_label_loss':loss.item()})
report={'purpose':'Training feasibility only, using arbitrary labels; not model quality or successful fine-tuning',
 'device_encoder':reader.device,'device_training_heads':'cpu','frames':len(frames),
 'total_parameters':sum(p.numel() for p in model.parameters()),
 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
 'trainable_modules':['ctc.ctc_lo','decoder.output_layer'],'steps':steps,
 'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
 'seconds_including_model_load_and_preprocessing':time.monotonic()-started,
 'production_weights_changed':False,'accuracy_measured':False}
Path('outputs/lipreading-studio/validation/pretrained-adaptation-feasibility.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
