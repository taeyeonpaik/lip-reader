"""Validation-selected beam/LM comparison; unchanged visual model weights."""
import argparse, datetime as dt, fcntl, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from studio.pipeline import LipReader, ROOT
from studio.adaptation import BASE_SHA256
from training.data import atomic_json,validate_splits
from training.experiment import Experiment
from training.metrics import summarize

CANDIDATES={'baseline':{'beam_size':5,'lm_weight':0.},'beam20':{'beam_size':20,'lm_weight':0.},'beam20-lm015':{'beam_size':20,'lm_weight':.15},'beam20-lm030':{'beam_size':20,'lm_weight':.3},'beam40-lm030':{'beam_size':40,'lm_weight':.3}}

class DecoderExperiment:
    def __init__(self,args):
        self.args=args;self.run=args.run.resolve();self.run.mkdir(parents=True,exist_ok=True)
        self.cache=args.cache.resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.rows=json.loads(args.manifest.read_text());validate_splits(self.rows)
        self.reader=LipReader();self.started=time.monotonic()
        config={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'base_sha256':BASE_SHA256,'lm':json.loads((ROOT/'models/english-lm/provenance.json').read_text()),'candidates':CANDIDATES,'pilot_validation_clips':100,'full_validation_clips':500,'finalists':2,'selection':'Lowest full-validation WER among two pilot finalists, CER no worse than baseline; no test tuning','test_note':'LRS2 test baseline already examined in previous experiments; test remains excluded from parameter selection','visual_weights_changed':False}
        with (ROOT/'models/autoavsr.pth').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==BASE_SHA256
        path=self.run/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Changed configuration on resume')
        atomic_json(path,config);self.config=config

    def status(self,stage,**data):
        data={'state':'running','stage':stage,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**data}
        atomic_json(self.run/'progress.json',data);print(json.dumps(data),flush=True)

    def features(self,row):
        if row.get('format')!='gray-npz':return Experiment.features(self,row)
        key=hashlib.sha256((row['sha256']+BASE_SHA256+':gray-center88-v1').encode()).hexdigest();path=self.cache/(key+'.npy')
        if path.exists():return torch.from_numpy(np.load(path,allow_pickle=False))
        with np.load(row['path'],allow_pickle=False) as data:frames=data['frames']
        if frames.dtype!=np.uint8 or frames.shape!=(row['declared_frames'],96,96):raise ValueError('Invalid grayscale crops')
        pixels=torch.from_numpy(frames[:,4:92,4:92].copy()).float()/255
        pixels=((pixels-.421)/.165).unsqueeze(0).unsqueeze(2).to(self.reader.device)
        with torch.inference_mode():
            x=self.reader.model.proj_encoder(self.reader.model.frontend(pixels));x,_=self.reader.model.encoder(x,None)
        features=x.squeeze(0).cpu().clone();tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.save(f,features.numpy(),allow_pickle=False)
        tmp.replace(path);return features

    def evaluate(self,rows,candidate,label,seed_records=None):
        destination=self.run/(label+'.json')
        if destination.exists():return json.loads(destination.read_text())
        params=CANDIDATES[candidate];self.reader.configure_decoding(**params,lm_path=ROOT/'models/english-lm' if params['lm_weight'] else None)
        log=self.run/(label+'.jsonl')
        if not log.exists() and seed_records:
            allowed={r['id'] for r in rows}
            log.write_text(''.join(json.dumps(r)+'\n' for r in seed_records if r['id'] in allowed))
        records=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        ids={r['id'] for r in records}
        self.status(label,completed=len(records),total=len(rows))
        with log.open('a') as f:
            for row in rows:
                if row['id'] in ids:continue
                prediction={'id':row['id'],'reference':row['text'],'prediction':''};start=time.monotonic()
                try:prediction['prediction']=self.reader.decode(self.features(row))
                except Exception as e:prediction['error']=f'{type(e).__name__}: {e}'
                prediction['seconds']=time.monotonic()-start
                records.append(prediction);f.write(json.dumps(prediction)+'\n');f.flush()
                if len(records)%25==0:self.status(label,completed=len(records),total=len(rows))
        result={'candidate':candidate,'decoding':params,'metrics':summarize(records),'mean_seconds':sum(r['seconds'] for r in records)/len(records),'predictions':records}
        atomic_json(destination,result);return result

    def execute(self):
        if (self.run/'result.json').exists():return
        val=[r for r in self.rows if r['split']=='validation'][:500];test=[r for r in self.rows if r['split']=='test']
        if len(val)!=500:raise ValueError('Expected 500 fixed LRS2 validation clips')
        pilot={c:self.evaluate(val[:100],c,'pilot-'+c) for c in CANDIDATES}
        eligible=[c for c in CANDIDATES if c!='baseline' and not pilot[c]['metrics']['failures']]
        finalists=sorted(eligible,key=lambda c:(pilot[c]['metrics']['wer'],pilot[c]['mean_seconds']))[:2]
        atomic_json(self.run/'finalists.json',{'candidates':finalists,'pilot_clips':100,'test_used':False})
        full={c:self.evaluate(val,c,'validation-'+c,seed_records=pilot[c]['predictions']) for c in ['baseline']+finalists}
        base=full['baseline']['metrics']
        if base['failures']:raise ValueError('Baseline failures must be resolved')
        qualified=[c for c in finalists if not full[c]['metrics']['failures'] and full[c]['metrics']['wer']<base['wer'] and full[c]['metrics']['cer']<=base['cer']]
        selected=min(qualified,key=lambda c:(full[c]['metrics']['wer'],full[c]['mean_seconds'])) if qualified else 'baseline'
        selection={'candidate':selected,'decoding':CANDIDATES[selected],'basis':'LRS2 validation only','validation':{c:x['metrics'] for c,x in full.items()},'test_used_for_selection':False}
        atomic_json(self.run/'selection.json',selection)
        baseline_test=self.evaluate(test,'baseline','baseline-test')
        selected_test=baseline_test if selected=='baseline' else self.evaluate(test,selected,'selected-test')
        result={'state':'completed','selection':selection,'baseline_test':baseline_test['metrics'],'selected_test':selected_test['metrics'],'baseline_mean_test_seconds':baseline_test['mean_seconds'],'selected_mean_test_seconds':selected_test['mean_seconds'],'visual_weights_changed':False,'app_profile_changed':False,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(self.run/'result.json',result);self.status('complete',state='completed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    with (a.run/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:DecoderExperiment(a).execute()
        except Exception as e:
            atomic_json(a.run/'progress.json',{'state':'failed','error':f'{type(e).__name__}: {e}'});raise
