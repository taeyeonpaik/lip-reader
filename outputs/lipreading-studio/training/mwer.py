"""Bounded N-best MWER experiment, with a matched conventional-training control.

Run from workspace root. No test data enters training, diagnosis, or selection.
"""
import argparse, datetime as dt, fcntl, hashlib, json, random, shutil, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
from torch.nn import functional as F
from studio.pipeline import ROOT, VENDOR, LipReader
from studio.adaptation import BASE_SHA256
from training.data import atomic_json, validate_splits
from training.decoding_experiment import DecoderExperiment
from training.metrics import summarize, normalize, edit_distance
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask

PREFIXES = ('decoder.decoders.4.', 'decoder.decoders.5.', 'decoder.after_norm.',
            'decoder.output_layer.', 'ctc.ctc_lo.')


def expected_word_error(scores, errors):
    """Centering by a constant preserves the expected-error gradient."""
    if scores.ndim != 1 or scores.shape != errors.shape or not len(scores):
        raise ValueError('Expected matching nonempty one-dimensional scores and errors')
    if not torch.isfinite(scores).all() or not torch.isfinite(errors).all():
        raise ValueError('Nonfinite N-best scores or errors')
    return (scores.softmax(0) * (errors - errors.mean())).sum()


def corpus_metrics(records):
    result = summarize(records)
    result['by_dataset'] = {d:summarize([r for r in records if r['dataset']==d]) for d in sorted({r['dataset'] for r in records})}
    result['macro_wer'] = sum(d['wer'] for d in result['by_dataset'].values())/len(result['by_dataset'])
    return result


def qualifies(metrics, baseline):
    return (not metrics['failures'] and baseline['macro_wer']-metrics['macro_wer']>=.005
            and metrics['cer']<=baseline['cer']
            and all(metrics['by_dataset'][d]['wer']<=b['wer']+.002 for d,b in baseline['by_dataset'].items()))


def restore(model, state):
    parameters = dict(model.named_parameters())
    expected = {n for n in parameters if n.startswith(PREFIXES)}
    if set(state)!=expected: raise ValueError('Unexpected or incomplete decoder adaptation')
    for n,t in state.items():
        if t.shape!=parameters[n].shape or not torch.isfinite(t).all(): raise ValueError('Invalid tensor '+n)
    with torch.no_grad():
        for n,t in state.items(): parameters[n].copy_(t)


class MWERExperiment:
    def __init__(self, args):
        self.args=args; self.run=args.run.resolve(); self.run.mkdir(parents=True,exist_ok=True)
        self.cache=args.cache.resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic()
        rows=json.loads(args.manifest.read_text());validate_splits(rows)
        if any(not r.get('dataset') for r in rows):raise ValueError('Dataset identity missing')
        self.rows=rows
        self.groups={k:[] for k in ['train','validation','audit','test']}
        for d in ['LRS2','LRS3-TED']:
            ordered=lambda split:sorted([r for r in rows if r['dataset']==d and r['split']==split], key=lambda r:hashlib.sha256(('mwer-v5-45:'+r['id']).encode()).hexdigest())
            self.groups['train']+=ordered('train')[:500]
            val=ordered('validation'); self.groups['validation']+=val[:200];self.groups['audit']+=val[200:600]
        self.groups['test']=[r for r in rows if r['split']=='test']
        assert len(self.groups['train'])==1000 and len(self.groups['validation'])==400 and len(self.groups['audit'])==800
        self.reader=LipReader(beam_size=20);self.model=self.reader.model
        with (ROOT/'models/autoavsr.pth').open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=BASE_SHA256:raise ValueError('Base hash mismatch')
        for n,p in self.model.named_parameters():p.requires_grad_(n.startswith(PREFIXES))
        self.params=[p for p in self.model.parameters() if p.requires_grad]
        self.original=self.state()
        self.sp=spm.SentencePieceProcessor(model_file=str(VENDOR/'spm/unigram/unigram5000.model'))
        self.units={s.split()[0]:int(s.split()[-1]) for s in (VENDOR/'spm/unigram/unigram5000_units.txt').read_text().splitlines()}
        self.config={'version':1,'base_sha256':BASE_SHA256,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                     'seed':45,'split_selection':'sha256(mwer-v5-45:+id), per dataset','counts':{k:len(v) for k,v in self.groups.items()},
                     'trainable_parameters':sum(p.numel() for p in self.params),'prefixes':list(PREFIXES),'beam_size':20,'nbest':10,
                     'epochs_per_arm':2,'accumulation':8,'learning_rate':1e-5,'weight_decay':.01,'gradient_clip':1.,
                     'arms':['ce','mwer'],'mwer_loss':'expected word-error count under softmax(.9 attention log probability + .1 CTC log probability) + .1 conventional loss',
                     'ce_loss':'.9 author label-smoothed attention loss + .1 CTC loss','dropout':'disabled in both arms; gradients enabled',
                     'candidate_refresh':'generated using epoch-start weights; fixed within epoch; no reference injected',
                     'oracle_gate_macro_gain':.02,'selection':'At least .005 absolute macro WER gain, pooled CER no worse, each dataset WER at most .002 worse. Select lowest macro WER before audit/test.',
                     'test_used_for_selection':False,'note':'Pilot only; existing validation/test previously examined. Not independent of original pretraining. No new tuning on audit/test.'}
        path=self.run/'config.json'
        if path.exists() and json.loads(path.read_text())!=self.config:raise ValueError('Config changed')
        atomic_json(path,self.config)
        atomic_json(self.run/'split-manifest.json',{k:[{'id':r['id'],'dataset':r['dataset'],'sha256':r['sha256']} for r in v] for k,v in self.groups.items()})

    def state(self):return {n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if n.startswith(PREFIXES)}
    def status(self,stage,**data):
        payload={'state':'running','stage':stage,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds_this_session':time.monotonic()-self.started,**data}
        atomic_json(self.run/'progress.json',payload);print(json.dumps(payload),flush=True)
    def features(self,row):
        if shutil.disk_usage(self.cache).free<6*1024**3:raise ValueError('Less than 6 GiB free')
        return DecoderExperiment.features(self,row)
    def tokens(self,text):return [self.units.get(p,self.units['<unk>']) for p in self.sp.encode(text,out_type=str)]
    def text(self,ids):return ''.join(self.reader.tokens[i] for i in ids if 0<i<len(self.reader.tokens)-1).replace('▁',' ').strip()

    def hypotheses(self,row):
        self.model.eval();x=self.features(row)
        with self.reader.lock,torch.inference_mode(): hyps=self.reader.search(x)
        candidates=[];seen=set()
        for h in hyps:
            seq=h.yseq.tolist()[1:]
            if self.model.eos in seq:seq=seq[:seq.index(self.model.eos)]
            key=tuple(seq)
            if key in seen:continue
            if any(i<=0 or i>=self.model.eos for i in seq):raise ValueError('Invalid candidate tokens')
            seen.add(key)
            candidates.append({'tokens':seq,'text':self.text(seq),'beam_score':float(h.score),
                               'word_errors':edit_distance(normalize(row['text']).split(),normalize(self.text(seq)).split())})
            if len(candidates)==10:break
        if not candidates:raise ValueError('Empty N-best list')
        return {'id':row['id'],'dataset':row['dataset'],'reference':row['text'],'candidates':candidates}

    def nbest(self,rows,label):
        path=self.run/(label+'.jsonl')
        records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        done={r['id'] for r in records}
        if len(done)!=len(records) or not done<={r['id'] for r in rows}:raise ValueError('Invalid N-best resume IDs')
        with path.open('a') as f:
            for row in rows:
                if row['id'] in done:continue
                record=self.hypotheses(row);records.append(record);f.write(json.dumps(record)+'\n');f.flush()
                if len(records)%25==0:self.status(label,completed=len(records),total=len(rows))
        return records

    def evaluate(self,rows,label):
        self.model.eval();path=self.run/(label+'.json')
        if path.exists():return json.loads(path.read_text())
        log=path.with_suffix('.jsonl');records=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else [];done={r['id'] for r in records}
        with log.open('a') as f:
            for row in rows:
                if row['id'] in done:continue
                start=time.monotonic();text=self.reader.decode(self.features(row))
                record={'id':row['id'],'dataset':row['dataset'],'reference':row['text'],'prediction':text,'seconds':time.monotonic()-start}
                records.append(record);f.write(json.dumps(record)+'\n');f.flush()
                if len(records)%25==0:self.status(label,completed=len(records),total=len(rows))
        result={'metrics':corpus_metrics(records),'predictions':records};atomic_json(path,result);return result

    def scores(self,x,sequences):
        labels=torch.nn.utils.rnn.pad_sequence([torch.tensor(s,dtype=torch.long) for s in sequences],batch_first=True,padding_value=-1)
        inp,out=add_sos_eos(labels,self.model.sos,self.model.eos,-1)
        memory=x.unsqueeze(0).expand(len(sequences),-1,-1)
        logits,_=self.model.decoder(inp,target_mask(inp,-1),memory,torch.ones(len(sequences),1,len(x),dtype=torch.bool))
        mask=out!=-1
        attention=(logits.log_softmax(-1).gather(-1,out.clamp_min(0).unsqueeze(-1)).squeeze(-1)*mask).sum(-1)
        lengths=torch.tensor([len(s) for s in sequences],dtype=torch.long)
        targets=torch.cat([torch.tensor(s,dtype=torch.long) for s in sequences])
        logp=self.model.ctc.log_softmax(x.unsqueeze(0)).transpose(0,1).expand(-1,len(sequences),-1).contiguous()
        ctc=-F.ctc_loss(logp,targets,torch.full((len(sequences),),len(x),dtype=torch.long),lengths,blank=0,reduction='none',zero_infinity=False)
        if not torch.isfinite(ctc).all():raise ValueError('Unalignable candidate CTC sequence')
        return .9*attention+.1*ctc

    def conventional_loss(self,x,text):
        ids=self.tokens(text);labels=torch.tensor([ids],dtype=torch.long)
        if len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))>len(x):raise ValueError('Reference cannot CTC-align')
        inp,out=add_sos_eos(labels,self.model.sos,self.model.eos,-1)
        memory=x.unsqueeze(0)
        ctc,_=self.model.ctc(memory,torch.tensor([len(x)]),labels)
        logits,_=self.model.decoder(inp,target_mask(inp,-1),memory,torch.ones(1,1,len(x),dtype=torch.bool))
        return .9*self.model.criterion(logits,out)+.1*ctc

    def train_arm(self,arm):
        restore(self.model,self.original);self.model.eval()
        optimizer=torch.optim.AdamW(self.params,lr=1e-5,weight_decay=.01)
        lastpath=self.run/(arm+'-last.pt');history=[];start_epoch=1;step=0
        if lastpath.exists():
            saved=torch.load(lastpath,map_location='cpu',weights_only=True);restore(self.model,saved['state']);optimizer.load_state_dict(saved['optimizer']);history=saved['history'];start_epoch=saved['epoch']+1;step=saved['step']
        for epoch in range(start_epoch,3):
            # Replay interrupted epoch from epoch-start weights and fixed candidates.
            for suffix in ['.json','.jsonl']:(self.run/f'{arm}-epoch-{epoch:02}-validation{suffix}').unlink(missing_ok=True)
            candidates={r['id']:r for r in self.nbest(self.groups['train'],f'{arm}-epoch-{epoch:02}-nbest')} if arm=='mwer' else {}
            random.seed(45+epoch);np.random.seed(45+epoch);torch.manual_seed(45+epoch)
            order=list(self.groups['train']);random.shuffle(order);total_loss=0.;total_ce=0.;tick=time.monotonic()
            for offset in range(0,len(order),8):
                batch=order[offset:offset+8];optimizer.zero_grad(set_to_none=True)
                for row in batch:
                    x=self.features(row);ce=self.conventional_loss(x,row['text']);loss=ce
                    if arm=='mwer':
                        cand=candidates[row['id']]['candidates'];scores=self.scores(x,[c['tokens'] for c in cand]);errors=torch.tensor([c['word_errors'] for c in cand],dtype=scores.dtype)
                        loss=expected_word_error(scores,errors)+.1*ce
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    total_loss+=float(loss.detach());total_ce+=float(ce.detach());(loss/len(batch)).backward()
                norm=torch.nn.utils.clip_grad_norm_(self.params,1.,error_if_nonfinite=True);optimizer.step();step+=1
                if step%10==0:self.status('training',arm=arm,epoch=epoch,completed=offset+len(batch),total=len(order),updates=step,gradient_norm=float(norm))
            report=self.evaluate(self.groups['validation'],f'{arm}-epoch-{epoch:02}-validation')
            history.append({'arm':arm,'epoch':epoch,'loss':total_loss/len(order),'conventional_loss':total_ce/len(order),'validation':report['metrics'],'updates':step,'seconds_training_and_validation':time.monotonic()-tick})
            bundle={'format':3,'base_sha256':BASE_SHA256,'prefixes':list(PREFIXES),'state':self.state(),'arm':arm,'epoch':epoch,'validation':report['metrics']}
            torch.save(bundle,self.run/f'{arm}-epoch-{epoch:02}.pt')
            temp=lastpath.with_suffix('.tmp');torch.save({**bundle,'optimizer':optimizer.state_dict(),'history':history,'step':step},temp);temp.replace(lastpath)
            atomic_json(self.run/(arm+'-history.json'),history)
        return history

    def execute(self):
        if (self.run/'result.json').exists():return
        restore(self.model,self.original)
        nbest=self.nbest(self.groups['validation'],'diagnostic-nbest')
        top=[{'id':r['id'],'dataset':r['dataset'],'reference':r['reference'],'prediction':r['candidates'][0]['text']} for r in nbest]
        oracle=[{**r,'prediction':min(n['candidates'],key=lambda c:c['word_errors'])['text']} for r,n in zip(top,nbest)]
        baseline=corpus_metrics(top);best=corpus_metrics(oracle)
        diagnostic={'baseline':baseline,'oracle':best,'macro_headroom':baseline['macro_wer']-best['macro_wer'],'oracle_is_not_achievable_accuracy':True,'gate_passed':baseline['macro_wer']-best['macro_wer']>=.02,'clips_with_better_candidate':sum(min(c['word_errors'] for c in r['candidates'])<r['candidates'][0]['word_errors'] for r in nbest)}
        atomic_json(self.run/'diagnostic.json',diagnostic)
        atomic_json(self.run/'baseline-validation.json',{'metrics':baseline,'predictions':top})
        if not diagnostic['gate_passed']:
            atomic_json(self.run/'result.json',{'state':'completed','outcome':'insufficient_candidate_headroom','diagnostic':diagnostic,'training_started':False,'production_changed':False});self.status('complete',state='completed');return
        history=self.train_arm('ce')+self.train_arm('mwer')
        eligible=[h for h in history if qualifies(h['validation'],baseline)]
        winner=min(eligible,key=lambda h:h['validation']['macro_wer']) if eligible else None
        selection={'selected':{'arm':winner['arm'],'epoch':winner['epoch']} if winner else 'pretrained','baseline':baseline,'candidates':history,'test_used':False,'fixed_before_audit':True}
        atomic_json(self.run/'selection.json',selection)
        result={'state':'completed','diagnostic':diagnostic,'selection':selection,'training_started':True,'production_changed':False,'total_updates':500,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        if winner:
            restore(self.model,self.original);base=self.evaluate(self.groups['audit'],'baseline-audit')
            bundle=torch.load(self.run/f"{winner['arm']}-epoch-{winner['epoch']:02}.pt",map_location='cpu',weights_only=True);restore(self.model,bundle['state'])
            candidate=self.evaluate(self.groups['audit'],'selected-audit')
            result['audit']={'baseline':base['metrics'],'candidate':candidate['metrics'],'consistent_gain':candidate['metrics']['macro_wer']<base['metrics']['macro_wer'] and candidate['metrics']['cer']<=base['metrics']['cer']}
            if result['audit']['consistent_gain']:
                restore(self.model,self.original);base_test=self.evaluate(self.groups['test'],'baseline-test')
                restore(self.model,bundle['state']);candidate_test=self.evaluate(self.groups['test'],'selected-test')
                result['test']={'baseline':base_test['metrics'],'candidate':candidate_test['metrics']}
        atomic_json(self.run/'result.json',result);self.status('complete',state='completed')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=Path('work/corpora/verified-manifest.json'));p.add_argument('--cache',type=Path,default=Path('work/overnight/features'));p.add_argument('--run',type=Path,default=ROOT/'training/runs/mwer-v5');return p

if __name__=='__main__':
    args=parser().parse_args();args.run.mkdir(parents=True,exist_ok=True)
    with (args.run/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:MWERExperiment(args).execute()
        except BaseException as e:
            atomic_json(args.run/'progress.json',{'state':'failed','error':f'{type(e).__name__}: {e}','updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()});raise
