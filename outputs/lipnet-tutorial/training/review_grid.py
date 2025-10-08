"""Independent post-run metric, split, reload and local runtime checks. No training."""
import json,hashlib,time,tempfile
from pathlib import Path
import numpy as np,torch
from tutorial.runtime import ROOT,GridReader,digest,read_rgb,prepare_upload
from tutorial.accelerated import greedy_decode
from tutorial.grammar import constrained_decode
from tutorial.live import OneTakeSession
RUN=ROOT/'runs/grid-return-v1';OUT=ROOT/'validation/grid'

def edits(a,b):
    a=a.lower().split();b=b.lower().split()
    d=np.zeros((len(a)+1,len(b)+1),dtype=np.int32);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
    return int(d[-1,-1]),len(a),a==b

def run():
    result=json.loads((RUN/'result.json').read_text());assert result['state']=='complete'
    cfg=json.loads((RUN/'config.json').read_text());manifest=json.loads((RUN/'split-manifest.json').read_text())
    assert [len(manifest[k]) for k in ['train','validation','test']]==[600,150,200]
    speakers={k:{r['id'].split('/')[0] for r in rows} for k,rows in manifest.items()}
    assert speakers=={'train':{'s1','s3'},'validation':{'s2'},'test':{'s4'}}
    hashes=[{r['sha256'] for r in manifest[k]} for k in ['train','validation','test']]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    # Independent GRID filename-to-transcript check; never used for inference.
    vocab=[dict(zip('blps',['bin','lay','place','set'])),dict(zip('bgrw',['blue','green','red','white'])),dict(zip('ab iw'.replace(' ',''),['at','by','in','with'])),{c:c for c in 'abcdefghijklmnopqrstuvxyz'},dict(zip('z123456789',['zero','one','two','three','four','five','six','seven','eight','nine'])),dict(zip('anps',['again','now','please','soon']))]
    for rows in manifest.values():
        for r in rows:assert ' '.join(v[c] for v,c in zip(vocab,r['id'].split('/')[1]))==r['reference']
    checked={};reports={};rng=np.random.default_rng(91019)
    for path in sorted(RUN.glob('*.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or 'predictions' not in data:continue
        expected=manifest['test' if path.stem.endswith('test') else 'validation'];records=data['predictions']
        assert [(r['id'],r['reference']) for r in records]==[(r['id'],r['reference']) for r in expected]
        checked[path.stem]={};reports[path.stem]=data
        for key in ['raw','constrained']:
            scores=[edits(r['reference'],r[key]) for r in records];errors=sum(s[0] for s in scores);words=sum(s[1] for s in scores)
            value={'clips':len(scores),'errors':errors,'words':words,'wer':errors/words,'exact':sum(s[2] for s in scores)}
            assert value==data[key],(path,key,value,data[key]);arr=np.array(scores,dtype=float)
            ix=rng.integers(0,len(arr),(10000,len(arr)));rates=arr[ix,0].sum(1)/arr[ix,1].sum(1)
            checked[path.stem][key]={**value,'clip_bootstrap_95_percent_interval':np.quantile(rates,[.025,.975]).tolist()}
    paired={};base=reports['baseline-validation']['predictions']
    for name,data in reports.items():
        if not name.startswith('epoch-'):continue
        differences=np.array([edits(a['reference'],b['constrained'])[0]-edits(a['reference'],a['constrained'])[0] for a,b in zip(base,data['predictions'])])/6
        ix=rng.integers(0,len(differences),(10000,len(differences)))
        paired[name]={'candidate_minus_author_wer':float(differences.mean()),'paired_clip_bootstrap_95_percent_interval':np.quantile(differences[ix].mean(1),[.025,.975]).tolist()}
    reader=GridReader();assert digest(ROOT/'models/tutorial-published.npz')==cfg['initialization_sha256']
    original={k:v.detach().cpu().clone() for k,v in reader.model.state_dict().items()}
    saved=torch.load(RUN/'last.pt',map_location='cpu',weights_only=True)
    assert saved['config_sha256']==digest(RUN/'config.json') and saved['epoch']==3
    changed=[k for k,v in saved['state_dict'].items() if not torch.equal(v,original[k])];assert changed
    assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
    pipeline=hashlib.sha256((digest(ROOT/'tutorial/runtime.py')+digest(ROOT/'models/face-template.json')).encode()).hexdigest()
    reload_checks=[]
    for model_name,state,expected in [('author',original,base),('last_epoch',saved['state_dict'],reports['epoch-03-validation']['predictions'])]:
        reader.model.load_state_dict(state,strict=True);reader.model.eval()
        for row,pred in zip(manifest['validation'][:3],expected[:3]):
            key=hashlib.sha256((row['sha256']+pipeline).encode()).hexdigest()
            with np.load(Path('work/grid-return/cache')/(key+'.npz')) as c:
                assert str(c['source'])==row['sha256'] and str(c['pipeline'])==pipeline;x=c['frames'].copy()
            logits=reader.logits(x);assert greedy_decode(logits[None])[0]==pred['raw'] and constrained_decode(logits)==pred['constrained']
            reload_checks.append({'model':model_name,'id':row['id'],'predictions_match':True})
    reader.model.load_state_dict(original);reader.model.eval()
    assert json.loads((ROOT/'models/grid-selection.json').read_text())['selected']=='author'
    runtime=[]
    for sample in ['bras9a','bbaf3s','lgwp4n']:
        source=ROOT/'samples'/sample/(sample+'.mpg')
        if not source.exists():continue
        frames=read_rgb(source);fixed=reader.recognize(frames,fixed=True)
        with tempfile.TemporaryDirectory() as folder:
            start=time.perf_counter();dest=Path(folder)/'silent.mp4';prepare_upload(source,dest);upload=reader.recognize(read_rgb(dest));total=time.perf_counter()-start
        live=OneTakeSession(lambda f:json.dumps(reader.recognize(f)))
        for i,frame in enumerate(frames):live.accept(frame,i/25)
        live.on_ended();assert live.wait_until_idle(60);snapshot=live.snapshot()
        assert len(snapshot['segments'])==1 and not snapshot['segments'][0].get('error')
        runtime.append({'sample':sample,'fixed':fixed,'upload':upload,'upload_total_seconds':total,'recorded_webcam':snapshot['segments'][0]})
    review={'status':'verified_negative_result','metrics':checked,'paired_validation_changes':paired,'uncertainty_note':'Clip bootstrap is conditional on one speaker per split; it does not estimate generalization uncertainty across speakers or independent recording sessions. A meaningful speaker bootstrap is impossible with one held-out speaker.','splits_verified':True,'filename_transcripts_verified':950,'cross_split_duplicate_hashes':False,'original_weights_sha256_unchanged':True,'changed_state_tensors':len(changed),'reload_checks':reload_checks,'active_model':'author','runtime':runtime,'physical_camera_tested':False,'pretraining_speaker_independence_certified':False,'additional_training_performed':False}
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'final-review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items() if k not in ['metrics','runtime','paired_validation_changes']}),flush=True)
if __name__=='__main__':run()
