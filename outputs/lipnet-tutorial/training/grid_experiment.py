"""Fixed local GRID adaptation; speaker-disjoint locally, original pretraining unknown."""
from pathlib import Path
import sys,json,time,random,hashlib,fcntl,shutil,datetime
import numpy as np,torch
from tutorial.runtime import ROOT,GridReader,read_rgb,aligned_input,digest
from tutorial.accelerated import VOCABULARY,greedy_decode,loss_ctc
from tutorial.grammar import constrained_decode,valid_sentence,word_error
RUN=ROOT/'runs/grid-return-v1';WORK=Path('work/grid-return');CACHE=WORK/'cache'

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def metrics(records,key):
    vals=[word_error(r['reference'],r[key]) for r in records];return {'clips':len(vals),'errors':sum(v['errors'] for v in vals),'words':sum(v['words'] for v in vals),'wer':sum(v['errors'] for v in vals)/sum(v['words'] for v in vals),'exact':sum(v['exact'] for v in vals)}
def checkpoint(model,epoch,config_hash):return {'format':'grid-tutorial-v1','epoch':epoch,'config_sha256':config_hash,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}

def run():
    RUN.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (RUN/'result.json').exists():return
    start=time.monotonic();random.seed(91);torch.manual_seed(91);torch.set_num_threads(4)
    def status(stage,**kwargs):
        v={'state':'running','stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs};write(RUN/'progress.json',v);print(json.dumps(v),flush=True)
    status('waiting_for_verified_GRID_downloads')
    wait=time.monotonic()
    while not (WORK/'acquisition.json').exists():
        if time.monotonic()-wait>3600:raise RuntimeError('GRID acquisition did not finish within one hour. Check acquire.log before resuming.')
        time.sleep(5)
    acquired=json.loads((WORK/'acquisition.json').read_text())
    status('verify_archive_checksums')
    for f in acquired['files']:
        with (WORK/f['key']).open('rb') as stream:assert 'md5:'+hashlib.file_digest(stream,'md5').hexdigest()==f['checksum']
    config={'seed':91,'train_speakers':['s1','s3'],'validation_speakers':['s2'],'test_speakers':['s4'],'train_clips_per_speaker':300,'validation_clips':150,'test_clips':200,'epochs':6,'minimum_epochs':3,'patience':2,'learning_rate':1e-5,'microbatch':2,'accumulation':4,'gradient_clip':5.,'maximum_training_seconds':7200,'beam':64,'selection':'constrained validation WER improves >=0.5 percentage points; raw WER must not worsen; best valid constrained WER, raw WER tie-break','initialization_sha256':digest(ROOT/'models/tutorial-published.npz'),'preprocessing':'automatic three-point face alignment to fixed GRID template; 75-frame resampling; tutorial grayscale/uint8 normalization','ctc':'actual target lengths, unlike tutorial padded-target loss; AMP float16 CNN with float32 LSTMs/CTC','trainable_parameters':8471924,'pretraining_speaker_independence_certified':False}
    cfg=RUN/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config,'Resume recipe mismatch'
    else:write(cfg,config)
    status('freeze_speaker_disjoint_manifest')
    config_hash=digest(cfg);rows={};manifest={};mapping={c:i+1 for i,c in enumerate(VOCABULARY)}
    for split,speakers,count in [('train',config['train_speakers'],300),('validation',['s2'],150),('test',['s4'],200)]:
        rows[split]=[]
        for speaker in speakers:
            base=WORK/'data'
            videos=sorted((base/speaker).glob('*.mpg'),key=lambda p:hashlib.sha256(('grid-v1-91:'+speaker+'/'+p.stem).encode()).hexdigest())
            if len(videos)<count:raise RuntimeError('Missing speaker '+speaker)
            for video in videos[:count]:
                if speaker=='s1':align=base/'alignments'/speaker/(video.stem+'.align')
                else:
                    found=list((WORK/'data').glob('**/'+speaker+'/**/'+video.stem+'.align'))
                    if len(found)!=1:raise RuntimeError('Alignment not unique for '+speaker+'/'+video.stem+': '+str(found))
                    align=found[0]
                text=' '.join(l.split()[-1] for l in align.read_text().splitlines() if l.strip() and l.split()[-1] not in ['sil','sp'])
                assert valid_sentence(text),text
                rows[split].append({'id':speaker+'/'+video.stem,'path':str(video),'reference':text,'sha256':digest(video),'alignment_sha256':digest(align)})
        manifest[split]=[{k:v for k,v in r.items() if k!='path'} for r in rows[split]]
    write(RUN/'split-manifest.json',manifest);write(RUN/'data-provenance.json',acquired)
    reader=GridReader();model=reader.model;initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};memory={}
    pipeline_hash=hashlib.sha256((digest(ROOT/'tutorial/runtime.py')+digest(ROOT/'models/face-template.json')).encode()).hexdigest()
    def inputs(row):
        key=hashlib.sha256((row['sha256']+pipeline_hash).encode()).hexdigest();path=CACHE/(key+'.npz')
        if row['id'] in memory:return memory[row['id']]
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                assert str(d['source'])==row['sha256'] and str(d['pipeline'])==pipeline_hash
                x=d['frames'].copy()
        else:
            if shutil.disk_usage(CACHE).free<4*1024**3:raise RuntimeError('Less than 4 GiB disk space')
            assert digest(row['path'])==row['sha256']
            x=aligned_input(read_rgb(row['path']));tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,frames=x,source=row['sha256'],pipeline=pipeline_hash)
            tmp.replace(path)
        assert x.shape==(75,46,140,1) and np.isfinite(x).all()
        memory[row['id']]=x;return x
    @torch.inference_mode()
    def evaluate(split,name):
        model.eval();records=[]
        for i,row in enumerate(rows[split]):
            logits=reader.logits(inputs(row));records.append({'id':row['id'],'reference':row['reference'],'raw':greedy_decode(logits[None])[0],'constrained':constrained_decode(logits,64)})
            if (i+1)%50==0:status('evaluation',report=name,completed=i+1,total=len(rows[split]))
        result={'raw':metrics(records,'raw'),'constrained':metrics(records,'constrained'),'predictions':records};write(RUN/(name+'.json'),result);return result
    status('prepare_aligned_training_inputs',total=750)
    for i,row in enumerate(rows['train']+rows['validation']):
        inputs(row)
        if (i+1)%25==0:status('prepare_aligned_training_inputs',completed=i+1,total=750)
    baseline_file=RUN/'baseline-validation.json'
    baseline=json.loads(baseline_file.read_text()) if baseline_file.exists() else evaluate('validation','baseline-validation')
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-5,eps=1e-7)
    amp=reader.device=='mps';scaler=torch.amp.GradScaler(reader.device,enabled=amp,init_scale=128)
    history=[];best_score=(baseline['constrained']['wer'],baseline['raw']['wer']);best_epoch=0;next_epoch=1;prior=0
    if (RUN/'last.pt').exists():
        saved=torch.load(RUN/'last.pt',map_location='cpu',weights_only=True);assert saved['config_sha256']==config_hash
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);history=saved['history'];best_score=tuple(saved['best_score']);best_epoch=saved['best_epoch'];next_epoch=saved['epoch']+1;prior=saved['training_seconds']
    train_start=time.monotonic();reason='maximum_epochs'
    for epoch in range(next_epoch,7):
        if prior+time.monotonic()-train_start>=7200:reason='training_time_budget';break
        order=list(rows['train']);random.Random(91+epoch).shuffle(order);torch.manual_seed(91+epoch);model.train();losses=[];opt.zero_grad(set_to_none=True)
        for index in range(0,len(order),2):
            batch=order[index:index+2];x=torch.from_numpy(np.stack([inputs(r) for r in batch])).to(reader.device)
            labels=[[mapping[c] for c in r['reference']] for r in batch];target=torch.tensor([c for text in labels for c in text]);lens=torch.tensor([len(t) for t in labels])
            with torch.autocast(reader.device,dtype=torch.float16,enabled=amp):logits=model(x)
            loss=loss_ctc(logits,target,lens)
            if not torch.isfinite(loss):raise ValueError('Nonfinite CTC loss')
            scaler.scale(loss/4).backward();losses.append(float(loss.detach()))
            if (index//2+1)%4==0 or index+2>=len(order):
                scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if (index//2+1)%25==0:status('training',epoch=epoch,batch=index//2+1,batches=300,mean_loss=float(np.mean(losses)))
        val=evaluate('validation',f'epoch-{epoch:02}-validation');score=(val['constrained']['wer'],val['raw']['wer'])
        qualifies=score[0]<=baseline['constrained']['wer']-.005 and score[1]<=baseline['raw']['wer']
        if qualifies and score<best_score:
            best_score=score;best_epoch=epoch;save(RUN/'best.pt',checkpoint(model,epoch,config_hash))
        entry={'epoch':epoch,'loss':float(np.mean(losses)),'raw_wer':score[1],'constrained_wer':score[0],'qualifies':qualifies};history.append(entry);write(RUN/'history.json',history)
        state=checkpoint(model,epoch,config_hash);state.update(optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=history,best_score=list(best_score),best_epoch=best_epoch,training_seconds=prior+time.monotonic()-train_start);save(RUN/'last.pt',state)
        status('epoch_complete',**entry)
        if epoch>=3 and epoch-best_epoch>=2:reason='validation_patience';break
    status('final_test',selected_epoch=best_epoch)
    model.load_state_dict(initial);base_test=evaluate('test','baseline-test')
    if best_epoch:
        selected=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(selected['state_dict']);selected_test=evaluate('test','selected-test')
        candidate_validation=evaluate('validation','selected-validation')
        changed=sum(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items());assert changed>0
        path=ROOT/'models/grid-adapted.pt';save(path,selected)
        reload_reader=GridReader(path)
        for row,pred in zip(rows['validation'][:3],candidate_validation['predictions'][:3]):
            assert torch.allclose(reader.logits(inputs(row)),reload_reader.logits(inputs(row)),atol=1e-5)
        write(RUN/'candidate-selection.json',{'selected':'adapted','sha256':digest(path),'epoch':best_epoch,'activation':'pending independent runtime and metric review'})
    else:
        selected_test=base_test;write(ROOT/'models/grid-selection.json',{'selected':'author','reason':'No adapted epoch passed validation'})
    checked=0
    for p in RUN.glob('*.json'):
        data=json.loads(p.read_text())
        if isinstance(data,dict) and 'predictions' in data:
            for key in ['raw','constrained']:assert metrics(data['predictions'],key)==data[key]
            checked+=1
    assert digest(ROOT/'models/tutorial-published.npz')==config['initialization_sha256']
    result={'state':'complete','epochs_completed':len(history),'selected':'adapted' if best_epoch else 'author','selected_epoch':best_epoch,'stop_reason':reason,'baseline_validation':{k:baseline[k] for k in ['raw','constrained']},'baseline_test':{k:base_test[k] for k in ['raw','constrained']},'selected_test':{k:selected_test[k] for k in ['raw','constrained']},'reports_recomputed':checked,'local_speaker_disjoint':True,'pretraining_speaker_independence_certified':False,'physical_camera_tested':False,'elapsed_seconds':time.monotonic()-start,'no_further_automatic_training':True}
    write(RUN/'result.json',result);write(RUN/'progress.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            (RUN/'failure.json').unlink(missing_ok=True);run()
        except BaseException as e:
            write(RUN/'failure.json',{'state':'failed','error':repr(e)});raise
