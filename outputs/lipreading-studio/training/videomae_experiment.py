"""Bounded transfer learning from generic VideoMAE, without lip-reading weights."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from studio.videomae import ROOT, TOKENS, FEATURE_VERSION, VideoFeatures, CTCHead, VideoMAEReader, labels, greedy, ctc_min_frames, sha256
from training.data import atomic_json, validate_splits
from training.metrics import summarize, normalize


def atomic_torch(path, value):
    tmp = Path(path).with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_crops(row):
    path = Path(row['path'])
    if row['format'] == 'gray-npz':
        # The verified release hashes the original NPY payload, before NPZ wrapping.
        with zipfile.ZipFile(path) as archive:
            digest = hashlib.sha256(archive.read('frames.npy')).hexdigest()
        if digest != row['sha256']:
            raise ValueError('Dataset NPY payload changed: ' + row['id'])
        with np.load(path, allow_pickle=False) as data:
            frames = data['frames'].copy()
    else:
        if sha256(path) != row['sha256']:
            raise ValueError('Dataset file changed: ' + row['id'])
        cap = cv2.VideoCapture(str(path)); images = []
        try:
            if abs(cap.get(cv2.CAP_PROP_FPS)-25) > .01:
                raise ValueError('Unexpected frame rate')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        frames = np.stack(images)
    if frames.dtype != np.uint8 or frames.shape[0] != row['declared_frames'] or frames.shape[1:3] != (96,96):
        raise ValueError('Invalid prepared mouth crops')
    return frames


def session(row):
    return row['id'].rsplit('/', 1)[0]


def metrics(records):
    out = summarize(records)
    out['by_dataset'] = {d: summarize([r for r in records if r['dataset']==d]) for d in sorted({r['dataset'] for r in records})}
    out['macro_wer'] = sum(x['wer'] for x in out['by_dataset'].values()) / len(out['by_dataset'])
    out['macro_cer'] = sum(x['cer'] for x in out['by_dataset'].values()) / len(out['by_dataset'])
    return out


def independent_metrics(records):
    # Separate full-matrix edit distance, used to verify serialized predictions.
    def distance(a,b):
        m=[[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): m[i][0]=i
        for j in range(len(b)+1): m[0][j]=j
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                m[i][j]=min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+(a[i-1]!=b[j-1]))
        return m[-1][-1]
    def base(rs):
        w=c=nw=nc=ex=failed=0
        for r in rs:
            a=normalize(r['reference']);b='' if r.get('error') else normalize(r['prediction'])
            w+=distance(a.split(),b.split());c+=distance(a,b);nw+=len(a.split());nc+=len(a);ex+=a==b;failed+=bool(r.get('error'))
        return dict(clips=len(rs),word_errors=w,reference_words=nw,wer=w/nw,character_errors=c,reference_characters=nc,cer=c/nc,exact_sentences=ex,failures=failed)
    out=base(records);out['by_dataset']={d:base([r for r in records if r['dataset']==d]) for d in sorted({r['dataset'] for r in records})}
    out['macro_wer']=sum(x['wer'] for x in out['by_dataset'].values())/len(out['by_dataset'])
    out['macro_cer']=sum(x['cer'] for x in out['by_dataset'].values())/len(out['by_dataset'])
    return out


class Experiment:
    def __init__(self,args):
        self.args=args;self.run=args.run.resolve();self.run.mkdir(parents=True,exist_ok=True)
        self.cache=args.cache.resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(66);random.seed(66);np.random.seed(66)
        self.provenance=json.loads((ROOT/'models/videomae-provenance.json').read_text())
        rows=json.loads(args.manifest.read_text());validate_splits(rows)
        if len(rows)!=80095:raise ValueError('Expected verified corpus manifest, including exclusion of unmatched LRS3 clip')
        groups={s:{session(r) for r in rows if r['split']==s} for s in ['train','validation','test']}
        if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError('Recording session crosses dataset splits')
        ordered=lambda rs:sorted(rs,key=lambda r:hashlib.sha256(('videomae-v6-66:'+r['id']).encode()).hexdigest())
        self.groups={k:[] for k in ['train','validation','heldout']};exclusions={}
        for dataset in ['LRS2','LRS3-TED']:
            train=[r for r in rows if r['dataset']==dataset and r['split']=='train']
            eligible=[r for r in train if 25<=r['declared_frames']<=150 and ctc_min_frames(labels(r['text']))<=r['declared_frames']]
            exclusions[dataset]={'training_total':len(train),'eligible_1_to_6_seconds_and_ctc_alignable':len(eligible)}
            self.groups['train']+=ordered(eligible)[:args.train_per_dataset]
            val=ordered([r for r in rows if r['dataset']==dataset and r['split']=='validation'])
            if dataset=='LRS3-TED':
                # Separate entire recording sessions for selection and final local audit.
                sessions=sorted({session(r) for r in val},key=lambda s:hashlib.sha256(('videomae-audit:'+s).encode()).hexdigest())
                audit_sessions=set(sessions[::2])
                selection=[r for r in val if session(r) not in audit_sessions]
                audit=[r for r in val if session(r) in audit_sessions]
            else:
                selection=val
                audit=ordered([r for r in rows if r['dataset']==dataset and r['split']=='test'])
            self.groups['validation']+=selection[:args.validation_per_dataset]
            self.groups['heldout']+=audit[:args.test_per_dataset]
        for name,size in [('train',2*args.train_per_dataset),('validation',2*args.validation_per_dataset),('heldout',2*args.test_per_dataset)]:
            if len(self.groups[name])!=size:raise ValueError('Insufficient '+name+' rows')
        held_sessions={session(r) for r in self.groups['heldout']}
        assert not held_sessions & {session(r) for r in self.groups['validation']}
        self.config={'experiment':'videomae-v6','smoke_test':args.smoke_test,'seed':66,'feature_version':FEATURE_VERSION,'encoder_sha256':self.provenance['encoder_sha256'],
            'manifest_sha256':sha256(args.manifest),'counts':{k:len(v) for k,v in self.groups.items()},'head_config':{'hidden':256,'layers':3,'dropout':.15},
            'max_epochs':args.epochs,'minimum_epochs_before_patience':12,'patience':8,'batch_size':32,'learning_rate':.001,'weight_decay':.0001,'gradient_clip':5.,
            'maximum_training_wall_seconds':args.budget_seconds,'selection':'lowest macro validation CER; macro WER breaks ties; heldout never used for selection',
            'encoder_training':'frozen generic-video encoder; all transcription layers initialized randomly and trained locally',
            'head_training':'3-layer bidirectional GRU + character CTC, CPU; dense 25fps slots reconstructed from 12.5fps tubelet features',
            'training_length_filter':exclusions,'feature_augmentation':'50% chance of masking one contiguous span of up to 3 feature steps per training example',
            'heldout_description':'200 LRS2 official-test samples plus 200 local LRS3 trainval audit samples; not official LRS3 test; previously used corpora',
            'audio_used':False,'lipreading_pretrained_weights_used':False,'recording_split_overlap':0,'speaker_independence_verified':False}
        cfg_path=self.run/'config.json'
        if cfg_path.exists() and json.loads(cfg_path.read_text())!=self.config:raise ValueError('Resume configuration mismatch')
        atomic_json(cfg_path,self.config)
        atomic_json(self.run/'split-manifest.json',{k:[{f:r[f] for f in ['id','dataset','sha256','source_repository','source_revision','declared_frames']} for r in rs] for k,rs in self.groups.items()})
        self.head=CTCHead(**self.config['head_config']);self.initial={n:p.detach().clone() for n,p in self.head.named_parameters()}
        self.optimizer=torch.optim.AdamW(self.head.parameters(),lr=.001,weight_decay=.0001)
        self.encoder=None;self.memory={};self.history=[];self.best_score=(float('inf'),float('inf'));self.best_epoch=0;self.next_epoch=1;self.prior_seconds=0.
        if (self.run/'last.pt').exists():
            saved=torch.load(self.run/'last.pt',map_location='cpu',weights_only=True)
            self.check_saved(saved);self.head.load_state_dict(saved['state_dict']);self.optimizer.load_state_dict(saved['optimizer'])
            self.history=saved['history'];self.best_score=tuple(saved['best_score']);self.best_epoch=saved['best_epoch'];self.next_epoch=saved['epoch']+1;self.prior_seconds=saved['elapsed_seconds']

    def elapsed(self):return self.prior_seconds+time.monotonic()-self.start
    def status(self,stage,**data):
        obj={'state':'running','stage':stage,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds':self.elapsed(),**data}
        atomic_json(self.run/'progress.json',obj);print(json.dumps(obj),flush=True)
    def cache_path(self,row):return self.cache/(hashlib.sha256((FEATURE_VERSION+':'+row['id']+':'+row['sha256']).encode()).hexdigest()+'.npz')
    def feature(self,row):
        if row['id'] in self.memory:return self.memory[row['id']]
        path=self.cache_path(row)
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                if str(d['source_sha256'])!=row['sha256'] or str(d['encoder_sha256'])!=self.config['encoder_sha256'] or str(d['feature_version'])!=FEATURE_VERSION:
                    raise ValueError('Feature cache metadata mismatch')
                array=d['features'].copy()
            if array.shape!=(math.ceil(row['declared_frames']/2),384) or array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError('Corrupt feature cache')
            value=torch.from_numpy(array)
        else:
            if shutil.disk_usage(self.cache).free<5*1024**3:raise RuntimeError('Less than 5 GiB free disk space')
            if self.encoder is None:self.encoder=VideoFeatures()
            value=self.encoder(load_crops(row)).clone().float()
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez(f,features=value.numpy(),source_sha256=row['sha256'],encoder_sha256=self.config['encoder_sha256'],feature_version=FEATURE_VERSION)
            temp.replace(path)
        self.memory[row['id']]=value
        return value
    def batch(self,rows,augment=False,zeros=False):
        lengths=torch.tensor([r['declared_frames'] for r in rows],dtype=torch.long)
        features=torch.zeros(len(rows),max(math.ceil(int(n)/2) for n in lengths),384)
        targets=[]
        for i,r in enumerate(rows):
            f=self.feature(r);features[i,:len(f)]=f
            if augment and random.random()<.5:
                start=random.randrange(len(f));features[i,start:min(start+3,len(f))]=0
            targets.append(labels(r['text']))
        if zeros:features.zero_()
        return features,lengths,torch.tensor([v for t in targets for v in t]),torch.tensor([len(t) for t in targets])
    @torch.inference_mode()
    def evaluate(self,rows,name,zeros=False):
        self.head.eval();records=[]
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];x,lens,_,_=self.batch(batch,zeros=zeros)
            predictions=greedy(self.head(x,lens),lens)
            records.extend({'id':r['id'],'dataset':r['dataset'],'reference':r['text'],'prediction':p} for r,p in zip(batch,predictions))
        result={'metrics':metrics(records),'predictions':records}
        atomic_json(self.run/(name+'.json'),result)
        return result
    def checkpoint(self,epoch,extra=None):
        return {'format':'videomae-ctc-v1','tokens':TOKENS,'feature_version':FEATURE_VERSION,'encoder_sha256':self.config['encoder_sha256'],
            'head_config':self.config['head_config'],'state_dict':{k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()},'epoch':epoch,
            'config_sha256':sha256(self.run/'config.json'),**(extra or {})}
    def check_saved(self,saved):
        if saved.get('format')!='videomae-ctc-v1' or saved.get('tokens')!=TOKENS or saved.get('encoder_sha256')!=self.config['encoder_sha256'] or saved.get('config_sha256')!=sha256(self.run/'config.json'):
            raise ValueError('Incompatible resume checkpoint')
    def run_training(self):
        if (self.run/'result.json').exists():return json.loads((self.run/'result.json').read_text())
        if not (self.run/'initial.pt').exists():atomic_torch(self.run/'initial.pt',self.checkpoint(0))
        all_rows=self.groups['train']+self.groups['validation']
        for i,row in enumerate(all_rows):
            self.feature(row)
            if i%50==0 or i+1==len(all_rows):self.status('features',completed=i+1,total=len(all_rows))
        if self.encoder is not None:
            del self.encoder;self.encoder=None
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        if not (self.run/'untrained-validation.json').exists():self.evaluate(self.groups['validation'],'untrained-validation')
        stop='maximum_epochs'
        for epoch in range(self.next_epoch,self.args.epochs+1):
            if self.elapsed()>self.args.budget_seconds and self.history:
                stop='wall_time_budget';break
            random.seed(66+epoch);torch.manual_seed(66+epoch)
            self.head.train();rows=list(self.groups['train']);random.shuffle(rows)
            # Bucket locally, while preserving randomized batches, to limit padding.
            chunks=[sorted(rows[i:i+256],key=lambda r:r['declared_frames']) for i in range(0,len(rows),256)]
            batches=[chunk[i:i+32] for chunk in chunks for i in range(0,len(chunk),32)];random.shuffle(batches)
            losses=[];begun=time.monotonic();self.status('training',epoch=epoch,total_epochs=self.args.epochs,batches=len(batches))
            for step,batch in enumerate(batches):
                self.optimizer.zero_grad(set_to_none=True)
                x,lens,target,target_lens=self.batch(batch,augment=True)
                logits=self.head(x,lens)
                loss=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),target,lens,target_lens,blank=0,zero_infinity=False)
                if not torch.isfinite(loss):raise ValueError('Nonfinite CTC loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),5.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
                self.optimizer.step();losses.append(float(loss.detach()))
                if (step+1)%25==0:self.status('training',epoch=epoch,batch=step+1,batches=len(batches),mean_loss=float(np.mean(losses)))
            val=self.evaluate(self.groups['validation'],f'epoch-{epoch:02}-validation')['metrics']
            score=(val['macro_cer'],val['macro_wer']);improved=score<self.best_score
            if improved:
                self.best_score=score;self.best_epoch=epoch
                atomic_torch(self.run/'best.pt',self.checkpoint(epoch,{'validation':val}))
            entry={'epoch':epoch,'training_loss':float(np.mean(losses)),'seconds':time.monotonic()-begun,'validation':val,'best_epoch':self.best_epoch}
            self.history.append(entry);atomic_json(self.run/'history.json',self.history)
            atomic_torch(self.run/'last.pt',self.checkpoint(epoch,{'optimizer':self.optimizer.state_dict(),'history':self.history,'best_epoch':self.best_epoch,'best_score':list(self.best_score),'elapsed_seconds':self.elapsed()}))
            self.status('epoch_completed',**entry)
            if epoch>=12 and epoch-self.best_epoch>=8:
                stop='validation_patience';break
        self.status('final_evaluation',selected_epoch=self.best_epoch)
        best=torch.load(self.run/'best.pt',map_location='cpu',weights_only=True);self.check_saved(best);self.head.load_state_dict(best['state_dict'])
        validation=self.evaluate(self.groups['validation'],'selected-validation')
        heldout=self.evaluate(self.groups['heldout'],'selected-heldout')
        ablation=self.evaluate(self.groups['validation'],'zero-visual-validation',zeros=True)
        # The existing lip reader is an evaluation-only comparator. Its weights,
        # outputs and features never serve as training data for the new model.
        comparison={}
        if not self.args.smoke_test:
            from studio.pipeline import LipReader
            from training.inputs import load_pixels
            self.status('autoavsr_comparison',selected_epoch=self.best_epoch)
            comparator=LipReader(beam_size=5)
            for split in ['validation','heldout']:
                records=[]
                for i,row in enumerate(self.groups[split]):
                    with torch.inference_mode():
                        x=load_pixels(row,comparator.device)
                        x=comparator.model.proj_encoder(comparator.model.frontend(x))
                        x,_=comparator.model.encoder(x,None)
                        prediction=comparator.decode(x.squeeze(0).cpu())
                    records.append({'id':row['id'],'dataset':row['dataset'],'reference':row['text'],'prediction':prediction})
                    if (i+1)%50==0:self.status('autoavsr_comparison',split=split,completed=i+1,total=len(self.groups[split]))
                comparison[split]=metrics(records)
                atomic_json(self.run/('autoavsr-'+split+'.json'),{'metrics':comparison[split],'predictions':records})
            del comparator
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        # Independently recompute every report from its serialized predictions.
        checked=[]
        for p in self.run.glob('*.json'):
            obj=json.loads(p.read_text())
            if isinstance(obj,dict) and 'predictions' in obj:
                assert independent_metrics(obj['predictions'])==obj['metrics'],p
                checked.append(p.name)
        initial=torch.load(self.run/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
        changes={k:float((v-initial[k]).abs().max()) for k,v in best['state_dict'].items()}
        if not all(np.isfinite(v) for v in changes.values()) or not any(v>0 for v in changes.values()):raise ValueError('No valid training changes')
        # Reconstruct the deployment reader and match predictions on raw crops.
        reader=VideoMAEReader(self.run/'best.pt');reload_checks=[]
        for dataset in ['LRS2','LRS3-TED']:
            for row in [r for r in self.groups['validation'] if r['dataset']==dataset][:2]:
                pred=reader.recognize(load_crops(row));expected=next(r['prediction'] for r in validation['predictions'] if r['id']==row['id'])
                if pred!=expected:raise ValueError('Reloaded prediction mismatch: '+row['id'])
                reload_checks.append({'id':row['id'],'prediction':pred,'matches':True})
        from studio.pipeline import track_mouth,prepare_video,frame_chunks
        from studio.live import LiveSession
        import tempfile
        example=ROOT/'examples/speaker.mp4'
        # Use the existing speaking-person example when its filename differs.
        if not example.exists():example=next(p for p in (ROOT/'examples').glob('*.mp4'))
        throughput={}
        with tempfile.TemporaryDirectory(prefix='videomae-review-') as tmp:
            silent=Path(tmp)/'silent.mp4';prepare_video(example,silent)
            frames=next(frame_chunks(silent,seconds=8))[1]
            t=time.perf_counter();crop,coverage=track_mouth(frames);prediction=reader.recognize(crop)
            throughput['upload_segment']={'video_seconds':len(frames)/25,'processing_seconds':time.perf_counter()-t,'prediction':prediction,'face_coverage':coverage,'includes_conversion':False}
            def process(rgb):return reader.recognize(track_mouth(rgb)[0])
            live=LiveSession(process)
            for i,frame in enumerate(frames):live.accept(frame,i/25)
            live.on_ended()
            if not live.wait_until_idle(180):raise RuntimeError('Webcam worker did not complete')
            snapshot=live.snapshot()
            if not snapshot['segments'] or any(r.get('error') for r in snapshot['segments']):raise ValueError('Webcam processing failure')
            throughput['webcam_recorded_frames']=snapshot;throughput['physical_camera_tested']=False
        assert sha256(ROOT/'models/videomae-small-kinetics.pt')==self.config['encoder_sha256']
        audit={'state':'passed','reports_recomputed':sorted(checked),'checkpoint_sha256':sha256(self.run/'best.pt'),'changed_tensors':changes,'reload_predictions':reload_checks,'throughput':throughput,'encoder_unchanged':True,'audio_used':False}
        atomic_json(self.run/'verification.json',audit)
        if not self.args.smoke_test:
            atomic_json(ROOT/'validation/videomae-completion.json',audit)
        # Publish as an explicitly experimental selectable model; keep Auto-AVSR default.
        if not self.args.smoke_test:
            atomic_torch(ROOT/'models/videomae-ctc.pt',best)
        result={'state':'completed','completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stop_reason':stop,'epochs_completed':len(self.history),'selected_epoch':self.best_epoch,
            'elapsed_seconds':self.elapsed(),'counts':self.config['counts'],'trainable_parameters':sum(p.numel() for p in self.head.parameters()),
            'validation':validation['metrics'],'heldout':heldout['metrics'],'zero_visual_validation':ablation['metrics'],'autoavsr_comparison':comparison,'verification':'passed','app_default':'Auto-AVSR','experimental_model_ready':True,
            'notes':['Frozen generic Kinetics VideoMAE encoder; new character CTC transcription layers trained locally.','No official LRS3 test; LRS3 audit is a local trainval subset.','No accuracy improvement over Auto-AVSR is claimed without a matched comparison.','The experiment has finished under its declared bounds; no further automatic training.']}
        atomic_json(self.run/'result.json',result);atomic_json(self.run/'progress.json',result)
        if not self.args.smoke_test:
            # Serialization containers can differ; checksum the actual published file.
            atomic_json(ROOT/'models/videomae-selection.json',{'ready':True,'checkpoint':'videomae-ctc.pt','checkpoint_sha256':sha256(ROOT/'models/videomae-ctc.pt'),'experimental':True,'default':False,'run':'training/runs/videomae-v6'})
        print(json.dumps(result,indent=2),flush=True);return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('work/corpora/verified-manifest.json'))
    p.add_argument('--cache',type=Path,default=Path('work/videomae/features'))
    p.add_argument('--run',type=Path,default=ROOT/'training/runs/videomae-v6')
    p.add_argument('--train-per-dataset',type=int,default=2500)
    p.add_argument('--validation-per-dataset',type=int,default=200)
    p.add_argument('--test-per-dataset',type=int,default=200)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--budget-seconds',type=int,default=7200)
    p.add_argument('--smoke-test',action='store_true',help='Never publish a smoke-test model to the app')
    return p


def main():
    args=parser().parse_args();args.run.mkdir(parents=True,exist_ok=True)
    with (args.run/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (args.run/'failure.json').unlink(missing_ok=True)
        try:Experiment(args).run_training()
        except BaseException as e:
            atomic_json(args.run/'failure.json',{'state':'failed','error':type(e).__name__+': '+str(e),'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            raise


if __name__=='__main__':main()
