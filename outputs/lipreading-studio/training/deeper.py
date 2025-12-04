"""Fine-tune the last two visual encoder and two text decoder blocks.

Only frozen prefix activations are cached. Trainable visual blocks run afresh on
all examples, retaining their gradient connection through the attention decoder.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import sentencepiece as spm
import torch

from studio.pipeline import ROOT, VENDOR, LipReader
from studio.adaptation import BASE_SHA256
from training.data import atomic_json, validate_splits
from training.metrics import summarize
from training.inputs import load_pixels
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask

PREFIXES = ('encoder.encoders.10.', 'encoder.encoders.11.', 'encoder.after_norm.',
            'decoder.decoders.4.', 'decoder.decoders.5.', 'decoder.after_norm.',
            'decoder.output_layer.', 'ctc.ctc_lo.')


def selected_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(PREFIXES)}


def restore(model, state):
    params = dict(model.named_parameters())
    expected = {n for n in params if n.startswith(PREFIXES)}
    if set(state) != expected: raise ValueError('Incomplete or unexpected adapted parameters')
    for n,t in state.items():
        if params[n].shape != t.shape or not torch.isfinite(t).all():
            raise ValueError('Invalid adapted tensor '+n)
    with torch.no_grad():
        for n,t in state.items(): params[n].copy_(t)


class Deeper:
    def __init__(self, args):
        self.args=args; self.run=args.run.resolve(); self.run.mkdir(parents=True,exist_ok=True)
        self.cache=args.cache.resolve(); self.cache.mkdir(parents=True,exist_ok=True)
        self.rows=json.loads(args.manifest.read_text()); validate_splits(self.rows)
        self.splits={s:[r for r in self.rows if r['split']==s] for s in ['train','validation','test']}
        with (ROOT/'models/autoavsr.pth').open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=BASE_SHA256:
                raise ValueError('Original checkpoint hash mismatch')
        self.reader=LipReader(device=args.device,beam_size=args.beam_size); self.model=self.reader.model
        self.original=selected_state(self.model)
        for name,p in self.model.named_parameters(): p.requires_grad_(name.startswith(PREFIXES))
        self.visual=[p for n,p in self.model.named_parameters() if n.startswith('encoder.') and p.requires_grad]
        self.language=[p for n,p in self.model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
        self.optimizer=torch.optim.AdamW([{'params':self.visual,'lr':args.lr,'peak_lr':args.lr},
                                         {'params':self.language,'lr':args.lr/3,'peak_lr':args.lr/3}],weight_decay=.01)
        self.sp=spm.SentencePieceProcessor(model_file=str(VENDOR/'spm/unigram/unigram5000.model'))
        self.units={s.split()[0]:int(s.split()[-1]) for s in (VENDOR/'spm/unigram/unigram5000_units.txt').read_text().splitlines()}
        self.config={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                     'base_sha256':BASE_SHA256,'prefixes':list(PREFIXES),'learning_rate_visual':args.lr,
                     'learning_rate_decoder':args.lr/3,'epochs':args.epochs,'minimum_epochs':args.min_epochs,
                     'patience':args.patience,'accumulation':args.accumulation,'microbatch':args.microbatch,'seed':args.seed,
                     'trainable_parameters':sum(p.numel() for p in self.visual+self.language),
                     'splits':{s:len(v) for s,v in self.splits.items()},'beam_size':args.beam_size,
                     'regularization':'0.1 dropout in unfrozen blocks; frozen batchnorm statistics; optional 4-frame feature mask',
                     'feature_mask_probability':args.mask_probability,'warmup_epochs':1,
                     'schedule':'one-epoch linear warmup then cosine decay to 10% peak',
                     'selection':'best validation WER with CER no worse than baseline; require 0.5 percentage point WER gain',
                     'test_used_for_selection':False,'test_note':'Official test baseline was examined in v1; never used for gradients or hyperparameter selection'}
        if args.balanced_datasets:
            if any(not r.get('dataset') for r in self.rows):raise ValueError('Dataset identities are required')
            self.config['dataset_counts']={d:{s:sum(r['dataset']==d and r['split']==s for r in self.rows) for s in self.splits} for d in sorted({r['dataset'] for r in self.rows})}
            if any(not counts['validation'] for counts in self.config['dataset_counts'].values()):raise ValueError('Each dataset needs validation data')
            self.config['balanced_datasets']=True
            self.config['selection']='Lowest mean of per-dataset validation WER, with no per-dataset WER or CER regression versus baseline; require 0.5 percentage point mean WER gain'
            self.config['test_note']='LRS2 official test only, previously examined. LRS3 trainval validation is held out of local training, not original pretraining; no official LRS3 test included.'
        path=self.run/'config.json'
        if path.exists() and json.loads(path.read_text())!=self.config:raise ValueError('Configuration changed on resume')
        atomic_json(path,self.config)
        self.started=time.monotonic()

    def status(self,stage,**data):
        atomic_json(self.run/'progress.json',{'state':'running','stage':stage,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds_this_session':time.monotonic()-self.started,**data})
        print(stage,json.dumps(data),flush=True)

    def prefix(self,row):
        version=':first10-gray-center88-v1' if row.get('format')=='gray-npz' else ':first10-center88-v1'
        key=hashlib.sha256((row['sha256']+BASE_SHA256+version).encode()).hexdigest()
        path=self.cache/(key+'.npy')
        if path.exists():return torch.from_numpy(np.load(path,allow_pickle=False))
        if shutil.disk_usage(self.cache).free<8*1024**3:raise ValueError('Less than 8 GiB free; stop before filling disk')
        pixels=load_pixels(row,self.reader.device)
        with torch.no_grad():
            x=self.model.proj_encoder(self.model.frontend(pixels))
            x=self.model.encoder.embed(x)
            for block in self.model.encoder.encoders[:10]:x,_=block(x,None)
            output=x[0].squeeze(0).cpu()
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.save(f,output.numpy(),allow_pickle=False)
        tmp.replace(path)
        return output

    def encode_suffix(self,features,augment=False):
        x=features.unsqueeze(0).to(self.reader.device)
        if augment and random.random()<self.args.mask_probability and x.shape[1]>20:
            # Feature masking is training-only; it never changes evaluation videos.
            start=random.randrange(x.shape[1]-4)
            x=x.clone();x[:,start:start+4]=0
        with torch.no_grad(): _,pos=self.model.encoder.embed(torch.zeros_like(x))
        for block in self.model.encoder.encoders[10:]: (x,pos),_=block((x,pos),None)
        return self.model.encoder.after_norm(x).cpu()

    def mode(self,training):
        self.model.eval()
        if training:
            for block in list(self.model.encoder.encoders[10:])+list(self.model.decoder.decoders[4:]):block.train()
            self.model.ctc.train()
            for module in self.model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

    def loss_batch(self,rows):
        features=[self.prefix(row) for row in rows]
        ilens=torch.tensor([len(f) for f in features])
        x=torch.nn.utils.rnn.pad_sequence(features,batch_first=True).to(self.reader.device)
        for i,length in enumerate(ilens.tolist()):
            if length>20 and random.random()<self.args.mask_probability:
                start=random.randrange(length-4);x[i,start:start+4]=0
        valid=torch.arange(x.shape[1],device=x.device)[None,:]<ilens.to(x.device)[:,None]
        memory_mask=valid.unsqueeze(1)
        with torch.no_grad():_,pos=self.model.encoder.embed(torch.zeros_like(x))
        for block in self.model.encoder.encoders[10:]: (x,pos),_=block((x,pos),memory_mask)
        x=self.model.encoder.after_norm(x).cpu()
        targets=[]
        for row,length in zip(rows,ilens.tolist()):
            ids=[self.units.get(piece,self.units['<unk>']) for piece in self.sp.encode(row['text'],out_type=str)]
            if len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))>length:raise ValueError('CTC reference too long')
            targets.append(torch.tensor(ids))
        labels=torch.nn.utils.rnn.pad_sequence(targets,batch_first=True,padding_value=self.model.ignore_id)
        inp,out=add_sos_eos(labels,self.model.sos,self.model.eos,self.model.ignore_id)
        ctc,_=self.model.ctc(x,ilens,labels)
        pred,_=self.model.decoder(inp,target_mask(inp,self.model.ignore_id),x,memory_mask.cpu())
        return .1*ctc+.9*self.model.criterion(pred,out)

    def loss(self,row):
        return self.loss_batch([row])

    def evaluate(self,split,name):
        self.mode(False)
        target=self.run/(name+'.json')
        if target.exists():return json.loads(target.read_text())['metrics']
        log=self.run/(name+'.jsonl')
        records=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        done={r['id'] for r in records}
        with log.open('a') as f:
            for row in self.splits[split]:
                if row['id'] in done:continue
                result={'id':row['id'],'reference':row['text'],'prediction':''}
                if self.args.balanced_datasets:result['dataset']=row['dataset']
                tick=time.monotonic()
                try:
                    with torch.no_grad():x=self.encode_suffix(self.prefix(row)).squeeze(0)
                    result['prediction']=self.reader.decode(x)
                except Exception as e:result['error']=f'{type(e).__name__}: {e}'
                result['seconds']=time.monotonic()-tick
                records.append(result);f.write(json.dumps(result)+'\n');f.flush()
                if len(records)%50==0:self.status(name,completed=len(records),total=len(self.splits[split]))
        metrics=summarize(records)
        if self.args.balanced_datasets:
            metrics['by_dataset']={d:summarize([r for r in records if r['dataset']==d]) for d in sorted({r['dataset'] for r in records})}
            metrics['macro_wer']=sum(m['wer'] for m in metrics['by_dataset'].values())/len(metrics['by_dataset'])
        atomic_json(target,{'metrics':metrics,'predictions':records,'split':split,'prepared_mouth_crops':True})
        return metrics

    def execute(self):
        if (self.run/'result.json').exists():return
        baseline=self.evaluate('validation','baseline-validation')
        if baseline['failures']:raise ValueError('Resolve baseline evaluation failures before training')
        failed=[]
        self.mode(False)
        for i,row in enumerate(self.splits['train'],1):
            try:self.prefix(row)
            except Exception as e:failed.append({'id':row['id'],'error':str(e)})
            if i%100==0:self.status('prepare-training',completed=i,total=len(self.splits['train']),failures=len(failed))
        atomic_json(self.run/'preprocessing-failures.json',failed)
        if failed:raise ValueError('Resolve training preprocessing failures before proceeding')
        steps_epoch=math.ceil(len(self.splits['train'])/self.args.accumulation)
        checkpoint=self.run/'last.pt';history=[];first=1;best=self.selection_score(baseline);best_epoch=0;step=0
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            restore(self.model,saved['state']);self.optimizer.load_state_dict(saved['optimizer'])
            history=saved['history'];first=saved['epoch']+1;best=saved['best_wer'];best_epoch=saved['best_epoch'];step=saved['step']
        for epoch in range(first,self.args.epochs+1):
            if epoch>self.args.min_epochs and epoch-1-best_epoch>=self.args.patience:break
            # An interrupted epoch is replayed from its preceding checkpoint.
            # Discard that epoch's partial predictions to avoid mixing weights.
            for suffix in ('.json','.jsonl'):
                (self.run/(f'epoch-{epoch:02}-validation'+suffix)).unlink(missing_ok=True)
            random.seed(self.args.seed+epoch);np.random.seed(self.args.seed+epoch);torch.manual_seed(self.args.seed+epoch)
            self.mode(True);order=list(self.splits['train']);random.shuffle(order)
            # Sort small shuffled windows by length to limit padding, while
            # preserving a different deterministic order for each epoch.
            order=[r for start in range(0,len(order),128) for r in sorted(order[start:start+128],key=lambda r:r['declared_frames'])]
            loss_sum=0.;tick=time.monotonic()
            for offset in range(0,len(order),self.args.accumulation):
                progress=step/max(1,steps_epoch)
                scale=(step+1)/steps_epoch if progress<1 else .1+.9*.5*(1+math.cos(math.pi*(progress-1)/max(1,self.args.epochs-1)))
                for group in self.optimizer.param_groups:group['lr']=group['peak_lr']*scale
                self.optimizer.zero_grad(set_to_none=True)
                group=order[offset:offset+self.args.accumulation]
                for start in range(0,len(group),self.args.microbatch):
                    microbatch=group[start:start+self.args.microbatch]
                    loss=self.loss_batch(microbatch)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss_sum+=float(loss.detach())*len(microbatch);(loss*len(microbatch)/len(group)).backward()
                norm=torch.nn.utils.clip_grad_norm_(self.visual+self.language,1.,error_if_nonfinite=True)
                self.optimizer.step();step+=1
                if step%25==0:self.status('training',epoch=epoch,epochs=self.args.epochs,completed=offset+len(group),total=len(order),updates=step,gradient_norm=float(norm))
            val=self.evaluate('validation',f'epoch-{epoch:02}-validation')
            history.append({'epoch':epoch,'training_loss':loss_sum/len(order),'validation':val,'seconds':time.monotonic()-tick,'updates':step})
            if self.qualifies(val,baseline) and self.selection_score(val)<best:
                best=self.selection_score(val);best_epoch=epoch
                torch.save({'format':2,'base_sha256':BASE_SHA256,'state':selected_state(self.model),'epoch':epoch,'validation':val},self.run/'best.pt')
            tmp=checkpoint.with_suffix('.tmp')
            torch.save({'state':selected_state(self.model),'optimizer':self.optimizer.state_dict(),'history':history,'epoch':epoch,'best_wer':best,'best_epoch':best_epoch,'step':step},tmp);tmp.replace(checkpoint)
            atomic_json(self.run/'history.json',history)
        selected=best_epoch>0 and self.selection_score(baseline)-best>=.005
        selection={'selected':'adapted' if selected else 'pretrained','best_epoch':best_epoch,'baseline_validation':baseline,'best_validation_wer':best,'test_used_for_selection':False}
        if self.args.balanced_datasets:
            selection['best_validation_macro_wer']=selection.pop('best_validation_wer')
        atomic_json(self.run/'selection.json',selection)
        restore(self.model,self.original)
        base_test=self.evaluate('test','baseline-test')
        adapted_test=None
        if best_epoch:
            candidate=torch.load(self.run/'best.pt',map_location='cpu',weights_only=True);restore(self.model,candidate['state'])
            adapted_test=self.evaluate('test','best-test')
        report={'state':'completed','config':self.config,'selection':selection,'epochs_completed':len(history),'updates':step,'baseline_test':base_test,'best_test':adapted_test,'production_weights_changed':False,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(self.run/'result.json',report);self.status('complete',state='completed')

    def selection_score(self,metrics):
        return metrics['macro_wer'] if self.args.balanced_datasets else metrics['wer']

    def qualifies(self,candidate,baseline):
        if candidate['failures']:return False
        if not self.args.balanced_datasets:return candidate['cer']<=baseline['cer']
        return all(candidate['by_dataset'][d]['wer']<=base['wer'] and candidate['by_dataset'][d]['cer']<=base['cer'] for d,base in baseline['by_dataset'].items())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--min-epochs',type=int,default=6);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=8);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--beam-size',type=int,default=5)
    p.add_argument('--balanced-datasets',action='store_true')
    p.add_argument('--seed',type=int,default=43);p.add_argument('--mask-probability',type=float,default=.3);p.add_argument('--device',default='auto')
    return p


if __name__=='__main__':
    args=parser().parse_args();args.run.mkdir(parents=True,exist_ok=True)
    with (args.run/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:Deeper(args).execute()
        except BaseException as e:
            atomic_json(args.run/'progress.json',{'state':'failed','error':f'{type(e).__name__}: {e}','updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()});raise
