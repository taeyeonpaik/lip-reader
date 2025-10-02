"""Independent metrics, source/split audit, checkpoint reload and runtime review."""
import hashlib,json,time,tempfile
from pathlib import Path
import numpy as np
from engines import ROOT,UP,Paper,paper_ctc_decode
from verify import distance
from inputs import inspect_upload
from live import OneTakeSession
from stable_spelling import stabilize_spelling
RUN=ROOT/'runs/paper-finetune-v1'
CACHE=Path('/private/tmp/lipnet-paper-finetune-v1/cache')
def read(name):return json.loads((RUN/name).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tally(rows,key='text'):
    errors=[distance(r['reference'],r[key]) for r in rows]
    words=[len(r['reference'].split()) for r in rows]
    return {'clips':len(rows),'errors':sum(errors),'words':sum(words),'wer':sum(errors)/sum(words),'exact':sum(r['reference']==r[key] for r in rows)}
def bootstrap(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    d=np.array([distance(x['reference'],y['text'])-distance(x['reference'],x['text']) for x,y in zip(a,b)])
    n=np.array([len(x['reference'].split()) for x in a]);samples=np.random.default_rng(193).integers(0,len(a),(5000,len(a)))
    rates=d[samples].sum(axis=1)/n[samples].sum(axis=1)
    return {'candidate_minus_baseline_wer':float(d.sum()/n.sum()),'conditional_clip_bootstrap_95':np.quantile(rates,[.025,.975]).tolist()}
def main():
    result=read('result.json');config=read('config.json');selected=read('selection-before-test.json');manifest=read('manifest.json');effective=read('effective-manifest.json');prep=read('preparation-report.json')
    upstream=json.loads((ROOT/'reports/source-manifest.json').read_text())
    for name,digest in upstream['sources'].items():assert sha(ROOT/name)==digest,name
    assert sha(UP/'evaluation/models/unseen-weights178.h5')==config['initial_weights_sha256']
    selected_epoch=selected['selected_epoch'];assert selected_epoch==min(result['state']['history'],key=lambda h:h['validation_wer'])['epoch']==4
    assert (RUN/'selection-before-test.json').stat().st_mtime<(RUN/'baseline-test.json').stat().st_mtime<(RUN/'candidate-test.json').stat().st_mtime
    groups={k:{r['id'].split('/')[0] for r in rows} for k,rows in manifest.items()}
    assert groups=={'train':{'s3','s4'},'validation':{'s1'},'test':{'s2'}}
    verified_sources=0;pipeline_hash=sha(ROOT/'engines.py')
    for split,rows in manifest.items():
        assert len({r['id'] for r in rows})==len(rows)
        for r in rows:
            assert sha(r['video'])==r['video_sha256'] and sha(r['align'])==r['alignment_sha256']
            ref=' '.join(line.split()[-1] for line in Path(r['align']).read_text().splitlines() if line.split() and line.split()[-1] not in ['sil','sp'])
            assert ref==r['reference'];verified_sources+=1
            if r['id'] not in prep['failures']:
                with np.load(CACHE/(r['id'].replace('/','_')+'.npz'),allow_pickle=False) as z:
                    assert str(z['video_sha256'])==r['video_sha256'] and str(z['pipeline_sha256'])==pipeline_hash
        expected=[r['id'] for r in rows if split!='train' or r['id'] not in prep['failures']]
        assert [r['id'] for r in effective[split]]==expected
    reports={}
    for path in sorted(RUN.glob('*validation.json'))+[RUN/'baseline-test.json',RUN/'candidate-test.json']:
        d=json.loads(path.read_text());rows=d['predictions']
        split='test' if path.name.endswith('-test.json') else 'validation'
        assert [(r['id'],r['reference']) for r in rows]==[(r['id'],r['reference']) for r in effective[split]]
        for key,field in [('text','metrics'),('raw','raw_metrics')]:
            recomputed=tally(rows,key)
            for k,v in recomputed.items():assert v==d[field][k],(path,k)
        reports[path.stem]={'metrics':tally(rows),'raw_metrics':tally(rows,'raw'),'failures':sum('error' in r for r in rows)}
    val_base=read('baseline-validation.json')['predictions'];val_candidate=read('selected-reload-validation.json')['predictions']
    test_base=read('baseline-test.json')['predictions'];test_candidate=read('candidate-test.json')['predictions']
    val_comparison=bootstrap(val_base,val_candidate);test_comparison=bootstrap(test_base,test_candidate)
    assert val_comparison['conditional_clip_bootstrap_95']==selected['comparison']['conditional_clip_bootstrap_95']
    assert test_comparison['conditional_clip_bootstrap_95']==result['test_comparison']['conditional_clip_bootstrap_95']
    base=stabilize_spelling(Paper());candidate=stabilize_spelling(Paper());candidate.model.load_weights(RUN/'best.weights.h5')
    # Keep original reports untouched; fix only the unspecified equal-score tie
    # order for BOTH models. Epoch choice remains frozen before this audit.
    stable={}
    for name in ['baseline-validation','selected-reload-validation','baseline-test','candidate-test']:
        saved=read(name+'.json')
        rows=[{**r,'text':base.spell.sentence(r['raw'])} for r in saved['predictions']]
        stable[name]={'metrics':tally(rows),'raw_metrics':tally(rows,'raw'),'predictions':rows,'preprocessing_failures':saved['preprocessing_failures']}
        (RUN/(name+'-stable.json')).write_text(json.dumps(stable[name],indent=2)+'\n')
    original_val_comparison=val_comparison;original_test_comparison=test_comparison
    val_comparison=bootstrap(stable['baseline-validation']['predictions'],stable['selected-reload-validation']['predictions'])
    test_comparison=bootstrap(stable['baseline-test']['predictions'],stable['candidate-test']['predictions'])
    differences=[float(np.max(np.abs(a-b))) for a,b in zip(base.model.get_weights(),candidate.model.get_weights())]
    assert sum(v>0 for v in differences)==20
    for a,b in zip(base.model.layers,candidate.model.layers):
        if a.name.startswith('batc'):assert all(np.array_equal(x,y) for x,y in zip(a.get_weights(),b.get_weights()))
    assert candidate.model.output_shape[-1]==28
    prediction_checks=[]
    for split in ['validation','test']:
        expected_base=stable['baseline-'+split]['predictions'];expected_candidate=stable['selected-reload-validation' if split=='validation' else 'candidate-test']['predictions']
        for i,(r,a,b) in enumerate(zip(effective[split],expected_base,expected_candidate)):
            assert r['id']==a['id']==b['id']
            if r.get('preprocessing_error'):
                assert a['text']==b['text']=='';continue
            with np.load(CACHE/(r['id'].replace('/','_')+'.npz'),allow_pickle=False) as z:x=z['x'][None].astype(np.float32)/255
            for model,saved in [(base,a),(candidate,b)]:
                p=model.model(x,training=False).numpy();decoded=paper_ctc_decode(p,[x.shape[1]],False,200)[0][0].numpy()[0]
                raw=model.labels_to_text(decoded);assert raw==saved['raw'] and model.spell.sentence(raw)==saved['text'],r['id']
            if (i+1)%25==0:print('Reload verified',split,i+1,flush=True)
        prediction_checks.append({'split':split,'clips':len(effective[split]),'baseline_and_candidate_match_saved':True})
    # Actual file upload preparation plus actual one-take worker, identical inputs.
    runtime={};source=UP/'evaluation/samples/id2_vcd_swwp2s.mpg'
    for name,model in [('baseline',base),('candidate',candidate)]:
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'upload.mpg';path.write_bytes(source.read_bytes());info=inspect_upload(path)
            uploads=[]
            for _ in range(2):r,p=model.predict(path);uploads.append(r)
        frames=model.Video().get_video_frames(str(source));captured=[]
        def callback(x):
            r,q=model.predict_frames(x);captured.append(q);return json.dumps(r)
        worker=OneTakeSession(callback)
        for i,frame in enumerate(frames):worker.accept(frame,i/25)
        worker.on_ended();assert worker.wait_until_idle(60)
        snap=worker.snapshot();assert len(snap['segments'])==1 and not snap['segments'][0].get('error')
        webcam=json.loads(snap['segments'][0]['text']);assert np.array_equal(p,captured[0])
        runtime[name]={'upload':uploads,'recorded_webcam':webcam,'upload_median_seconds':float(np.median([r['seconds'] for r in uploads])),'recorded_worker_probability_max_error':0.0,'duration_seconds':info['duration_seconds']}
    speed_ratio=runtime['candidate']['upload_median_seconds']/runtime['baseline']['upload_median_seconds']
    eligible=selected['qualifies_on_validation'] and val_comparison['candidate_minus_baseline_wer']<=-.005 and val_comparison['conditional_clip_bootstrap_95'][1]<0 and test_comparison['candidate_minus_baseline_wer']<=0 and speed_ratio<=1.25
    review={'reviewed_unix':time.time(),'selected_epoch':selected_epoch,'selected_before_test_verified':True,'upstream_hashes_verified':len(upstream['sources']),'source_videos_and_alignments_verified':verified_sources,'effective_counts':{k:len(v) for k,v in effective.items()},'excluded_training_clips':[r['id'] for r in manifest['train'] if r['id'] in prep['failures']],'preprocessing_failures':prep['failures'],'original_metrics':reports,'metrics':{k:{'metrics':v['metrics'],'raw_metrics':v['raw_metrics'],'failures':v['preprocessing_failures']} for k,v in stable.items()},'original_validation_comparison':original_val_comparison,'original_test_comparison':original_test_comparison,'spelling_tie_fix':'Alphabetical order among equal-probability candidates; same rule for both models, no reference input; epoch choice unchanged','validation_comparison':val_comparison,'test_comparison':test_comparison,'changed_weight_tensors':20,'batchnorm_weights_unchanged':True,'original_weight_sha256':config['initial_weights_sha256'],'candidate_sha256':sha(RUN/'best.weights.h5'),'vocabulary_classes':28,'prediction_reload_checks':prediction_checks,'runtime':runtime,'candidate_to_baseline_upload_time_ratio':speed_ratio,'eligible_for_activation':bool(eligible),'physical_camera_tested':False,'limitations':'One validation and one test speaker. Test paired confidence interval must be read alongside the point estimate; it is conditional on these clips. No general sub-10% or webcam-accuracy claim.'}
    (RUN/'independent-review.json').write_text(json.dumps(review,indent=2)+'\n');print('REVIEW COMPLETE',eligible,flush=True)
if __name__=='__main__':main()
