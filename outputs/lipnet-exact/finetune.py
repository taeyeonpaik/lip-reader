"""Bounded, resumable fine-tuning of the verified original paper model."""
import argparse,hashlib,json,os,time,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from engines import ROOT,UP,Paper,keras,tf,paper_ctc_batch_cost,paper_ctc_decode
from verify import text_from_id
from audit_reports import metrics

WORKSPACE=ROOT.parents[1]
RUN=ROOT/'runs/paper-finetune-v1'
CACHE=Path('/private/tmp/lipnet-paper-finetune-v1/cache')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def status(stage,**kw):
    value={'stage':stage,'updated_unix':time.time(),**kw};write(RUN/'progress.json',value);print(json.dumps(value),flush=True)

def recipe():
    return {'seed':193,'initial_weights_sha256':sha(UP/'evaluation/models/unseen-weights178.h5'),'model':'unchanged upstream model2','train_speakers':['s3','s4'],'train_per_speaker':600,'validation_speaker':'s1','validation_clips':150,'test_speaker':'s2','test_clips':200,'exclude_prior_exact_evaluation_clips':True,'learning_rate':1e-5,'adam_epsilon':1e-8,'microbatch':2,'accumulation':8,'effective_batch':16,'freeze_batch_normalization':True,'augmentation':'original horizontal flip0.5 and temporal jitter0.05','max_epochs':6,'min_epochs':3,'patience':2,'training_budget_seconds':7200,'decoder':'original CTC beam200 log epsilon1e-8 and original Spell','minimum_validation_absolute_gain':.005,'candidate_requires_validation_bootstrap_upper_below_zero':True,'cache_location':str(CACHE),'no_automatic_activation':True}

def manifest(config):
    data=Path('/private/tmp/lipnet-paper-finetune-v1/data');excluded=set()
    for name in ['tutorial','paper']:
        report=json.loads((ROOT/f'reports/{name}-verification.json').read_text())
        excluded.update(r['id'] for r in report['predictions'])
    result={}
    for split,speakers,count in [('train',['s3','s4'],600),('validation',['s1'],150),('test',['s2'],200)]:
        rows=[]
        for speaker in speakers:
            paths=[p for p in (data/speaker).glob('*.mpg') if speaker+'/'+p.name not in excluded]
            paths.sort(key=lambda p:hashlib.sha256(('paper-ft-v1-193:'+speaker+'/'+p.name).encode()).hexdigest())
            assert len(paths)>=count
            for p in paths[:count]:
                align=data/'alignments'/speaker/(p.stem+'.align')
                ref=' '.join(line.split()[-1] for line in align.read_text().splitlines() if line.split() and line.split()[-1] not in ['sil','sp'])
                assert ref==text_from_id(p.stem),(p,ref)
                rows.append({'id':speaker+'/'+p.name,'video':str(p.resolve()),'align':str(align.resolve()),'reference':ref,'video_sha256':sha(p),'alignment_sha256':sha(align)})
        result[split]=rows
    assert not set(r['id'].split('/')[0] for r in result['train']) & set(r['id'].split('/')[0] for r in result['validation']+result['test'])
    return result

def cache_path(row):return CACHE/(row['id'].replace('/','_')+'.npz')
def prepare(row):
    from lipnet.lipreading.videos import Video
    path=cache_path(row)
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as z:
                x=z['x'];valid=str(z['video_sha256'])==row['video_sha256'] and str(z['pipeline_sha256'])==sha(ROOT/'engines.py') and x.dtype==np.uint8 and x.shape[1:]==(100,50,3) and len(x) in [74,75]
            if valid:return {'id':row['id'],'cached':True,'valid':True}
        except (ValueError,OSError,KeyError):pass
    assert sha(row['video'])==row['video_sha256']
    try:
        v=Video(vtype='face',face_predictor_path=str(UP/'common/predictors/shape_predictor_68_face_landmarks.dat')).from_video(row['video'])
    except ValueError as error:
        # Original cropping can yield empty/ragged arrays. Keep its behavior;
        # report this clip as unusable instead of inventing a replacement crop.
        return {'id':row['id'],'valid':False,'error':'Original video preparation failed: '+str(error)}
    x=v.data
    if x.dtype!=np.uint8 or x.ndim!=4 or x.shape[1:]!=(100,50,3) or len(x) not in [74,75]:
        return {'id':row['id'],'valid':False,'error':f'Original mouth detector returned invalid shape/dtype: {x.shape}, {x.dtype}'}
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,x=x,video_sha256=row['video_sha256'],pipeline_sha256=sha(ROOT/'engines.py'))
    temp.replace(path);return {'id':row['id'],'cached':False,'valid':True}

def load(row):
    with np.load(cache_path(row),allow_pickle=False) as z:return z['x'].copy()

def evaluate(model,rows,name):
    records=[]
    for i,r in enumerate(rows):
        if r.get('preprocessing_error'):
            records.append({'id':r['id'],'reference':r['reference'],'raw':'','text':'','error':r['preprocessing_error']})
        else:
            x=load(r).astype(np.float32)[None]/255
            p=model.model(x,training=False).numpy();d=paper_ctc_decode(p,[x.shape[1]],False,200)[0][0].numpy()[0]
            raw=model.labels_to_text(d);text=model.spell.sentence(raw)
            records.append({'id':r['id'],'reference':r['reference'],'raw':raw,'text':text})
        if (i+1)%25==0:status('evaluation',name=name,completed=i+1,total=len(rows))
    result={'metrics':metrics(records),'raw_metrics':metrics(records,'raw'),'predictions':records,'preprocessing_failures':sum('error' in r for r in records)}
    write(RUN/(name+'.json'),result);return result

def paired(base,candidate):
    from audit_reports import edits
    assert [r['id'] for r in base]==[r['id'] for r in candidate]
    dif=np.array([edits(a['reference'],b['text'])-edits(a['reference'],a['text']) for a,b in zip(base,candidate)])
    words=np.array([len(r['reference'].split()) for r in base]);rng=np.random.default_rng(193)
    idx=rng.integers(0,len(base),(5000,len(base)));delta=dif[idx].sum(1)/words[idx].sum(1)
    return {'candidate_minus_baseline_wer':float(dif.sum()/words.sum()),'conditional_clip_bootstrap_95':np.quantile(delta,[.025,.975]).tolist(),'limitation':'One held-out speaker; clip resampling does not estimate generalization across new speakers.'}

def train_batch(rows,epoch,batch_index):
    from lipnet.lipreading.videos import Video,VideoAugmenter
    from lipnet.lipreading.aligns import Align
    from lipnet.lipreading.helpers import text_to_labels
    from lipnet.lipreading.curriculums import Curriculum
    np.random.seed(193+epoch*100000+batch_index)
    curriculum=Curriculum(lambda _:dict(sentence_length=-1,flip_probability=.5,jitter_probability=.05));curriculum.update(epoch,train=True)
    xs=[];ys=[];lens=[]
    for row in rows:
        v=Video(vtype='mouth').from_array(load(row).swapaxes(1,2))
        a=Align(32,text_to_labels).from_file(row['align']);v,a,_=curriculum.apply(v,a);v=VideoAugmenter.pad(v,75)
        xs.append(v.data);ys.append(a.padded_label);lens.append(a.label_length)
    return np.asarray(xs,np.float32)/255,np.asarray(ys,np.float32),np.full((len(xs),1),75,np.int32),np.asarray(lens,np.int32)[:,None]

def main(smoke=False):
    RUN.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'result.json').exists():print('Already complete; no new training.',flush=True);return
    config=recipe()
    if (RUN/'config.json').exists():assert json.loads((RUN/'config.json').read_text())==config,'Config changed; refusing resume'
    else:write(RUN/'config.json',config)
    keras.utils.set_random_seed(config['seed']);m=Paper('unseen')
    if smoke and not (RUN/'manifest.json').exists():
        base=Path('/private/tmp/lipnet-paper-finetune-v1/data');rows=[]
        for p in sorted((base/'s3').glob('*.mpg'))[:2]:
            a=base/'alignments/s3'/(p.stem+'.align')
            rows.append({'id':'s3/'+p.name,'video':str(p),'align':str(a),'reference':text_from_id(p.stem),'video_sha256':sha(p),'alignment_sha256':sha(a)})
        assert len(rows)==2;data={'train':rows}
    elif (RUN/'manifest.json').exists():data=json.loads((RUN/'manifest.json').read_text())
    else:
        status('verifying_manifest')
        data=manifest(config);write(RUN/'manifest.json',data)
    if smoke:
        for r in data['train'][:2]:assert prepare(r)['valid'],r['id']
    else:
        write(RUN/'data-quality-policy.json',{'reason':'Recover from original detector failure before any training epoch','train':'Exclude invalid original crops, record IDs; no replacement samples','validation_test':'Keep every selected clip; a preprocessing failure predicts empty text and counts all reference words as errors','abort_if_training_exclusions_exceed_fraction':.05,'architecture_preprocessing_decoder_and_hyperparameters_unchanged':True})
        allrows=data['train']+data['validation']+data['test'];status('preparing_original_mouth_crops',completed=0,total=len(allrows))
        failures={};cached=0
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,r in enumerate(pool.map(prepare,allrows)):
                if not r['valid']:failures[r['id']]=r['error']
                cached+=int(r.get('cached',False))
                if (i+1)%10==0 or i+1==len(allrows):
                    write(RUN/'preparation-report.json',{'completed':i+1,'total':len(allrows),'reused_caches':cached,'failures':failures})
                    status('preparing_original_mouth_crops',completed=i+1,total=len(allrows),last=r['id'],preprocessing_failures=len(failures),reused_caches=cached)
        for split,rows in data.items():
            data[split]=[{**r,**({'preprocessing_error':failures[r['id']]} if r['id'] in failures else {})} for r in rows if split!='train' or r['id'] not in failures]
        assert len(data['train'])>=.95*config['train_per_speaker']*len(config['train_speakers']),'Too many invalid training crops; review the pipeline before proceeding'
        effective=RUN/'effective-manifest.json'
        if (RUN/'state.json').exists():assert json.loads(effective.read_text())==data,'Effective dataset changed after training began'
        write(effective,data)
    for layer in m.model.layers:
        if isinstance(layer,keras.layers.BatchNormalization):layer.trainable=False
    opt=keras.optimizers.legacy.Adam(learning_rate=config['learning_rate'],epsilon=config['adam_epsilon'])
    variables=m.model.trainable_weights
    @tf.function(reduce_retracing=True)
    def gradient(x,y,n,l):
        with tf.GradientTape() as tape:
            p=m.model(x,training=True);loss=tf.reduce_mean(paper_ctc_batch_cost(y,p,n,l))
        grads=tape.gradient(loss,variables)
        tf.debugging.assert_all_finite(loss,'Non-finite CTC loss')
        for g in grads:tf.debugging.assert_all_finite(g,'Non-finite gradient')
        return loss,grads
    if smoke:
        before=[v.numpy().copy() for v in variables];x=train_batch(data['train'][:2],0,0);loss,grads=gradient(*x);opt.apply_gradients(zip(grads,variables))
        changed=sum(not np.array_equal(a,b.numpy()) for a,b in zip(before,variables));assert changed
        write(RUN/'smoke.json',{'loss':float(loss),'changed_trainable_tensors':changed,'frozen_bn_layers':3,'published_weights_unchanged':sha(m.weights)==config['initial_weights_sha256'],'training_batch_shape':list(x[0].shape),'completed_full_epoch':False});print('SMOKE PASSED',float(loss),flush=True);return
    baseline_path=RUN/'baseline-validation.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else evaluate(m,data['validation'],'baseline-validation')
    state_path=RUN/'state.json';state={'completed_epoch':0,'best_wer':baseline['metrics']['wer'],'best_epoch':0,'stale':0,'training_seconds':0,'history':[]}
    checkpoint=tf.train.Checkpoint(model=m.model,optimizer=opt)
    manager=tf.train.CheckpointManager(checkpoint,str(RUN/'resume'),max_to_keep=1)
    if state_path.exists():
        state=json.loads(state_path.read_text());assert manager.latest_checkpoint
        assert Path(manager.latest_checkpoint).name==f"ckpt-{state['completed_epoch']}", 'Incomplete checkpoint transaction requires manual review'
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    for epoch in range(state['completed_epoch']+1,config['max_epochs']+1):
        if state['training_seconds']>=config['training_budget_seconds']:break
        start=time.monotonic();order=np.random.default_rng(config['seed']+epoch).permutation(len(data['train']));accum=[tf.zeros_like(v) for v in variables];pending=0;pending_examples=0;losses=[]
        for batch_index,offset in enumerate(range(0,len(order),config['microbatch'])):
            rows=[data['train'][j] for j in order[offset:offset+config['microbatch']]];loss,grads=gradient(*train_batch(rows,epoch,batch_index));losses.append(float(loss));accum=[a+g*len(rows) for a,g in zip(accum,grads)];pending+=1;pending_examples+=len(rows)
            if pending==config['accumulation'] or offset+config['microbatch']>=len(order):
                opt.apply_gradients((a/pending_examples,v) for a,v in zip(accum,variables));accum=[tf.zeros_like(v) for v in variables];pending=0;pending_examples=0
            if (batch_index+1)%25==0:status('training',epoch=epoch,max_epochs=config['max_epochs'],microbatch=batch_index+1,total_microbatches=(len(order)+config['microbatch']-1)//config['microbatch'],mean_loss=float(np.mean(losses)))
        elapsed=time.monotonic()-start;validation=evaluate(m,data['validation'],f'epoch-{epoch}-validation');wer=validation['metrics']['wer']
        if wer<state['best_wer']:
            state['best_wer']=wer;state['best_epoch']=epoch;state['stale']=0;m.model.save_weights(RUN/'best.weights.h5')
        else:state['stale']+=1
        state['completed_epoch']=epoch;state['training_seconds']+=elapsed;state['history'].append({'epoch':epoch,'mean_loss':float(np.mean(losses)),'validation_wer':wer,'training_seconds':elapsed})
        manager.save(checkpoint_number=epoch);write(state_path,state);status('epoch_complete',**state)
        if epoch>=config['min_epochs'] and state['stale']>=config['patience']:break
    selected=state['best_epoch'];comparison=None;qualifies=False
    if selected:
        m.model.load_weights(RUN/'best.weights.h5');validation=evaluate(m,data['validation'],'selected-reload-validation')
        original=json.loads((RUN/f'epoch-{selected}-validation.json').read_text());assert [r['text'] for r in validation['predictions']]==[r['text'] for r in original['predictions']]
        comparison=paired(baseline['predictions'],validation['predictions']);qualifies=comparison['candidate_minus_baseline_wer']<=-config['minimum_validation_absolute_gain'] and comparison['conditional_clip_bootstrap_95'][1]<0
    selection={'selected_epoch':selected,'qualifies_on_validation':qualifies,'comparison':comparison,'test_not_yet_evaluated':True};write(RUN/'selection-before-test.json',selection)
    # Test only after selection is written and fixed; it never chooses an epoch.
    base=Paper('unseen');base_test=evaluate(base,data['test'],'baseline-test')
    candidate_test=evaluate(m,data['test'],'candidate-test') if selected else None
    changed=sum(not np.array_equal(a,b) for a,b in zip(base.model.get_weights(),m.model.get_weights())) if selected else 0
    if selected:assert changed>0
    result={'config':config,'state':state,'selection':selection,'baseline_test':base_test['metrics'],'candidate_test':candidate_test['metrics'] if candidate_test else None,'test_comparison':paired(base_test['predictions'],candidate_test['predictions']) if candidate_test else None,'changed_model_tensors':changed,'published_weights_unchanged':sha(m.weights)==config['initial_weights_sha256'],'activation':'pending independent review; app remains published baseline','physical_camera_tested':False}
    write(RUN/'result.json',result);status('complete',selected_epoch=selected,qualifies_on_validation=qualifies)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a.smoke)
    except Exception as e:status('failed',error=repr(e));raise
