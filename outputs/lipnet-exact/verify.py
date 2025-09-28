"""Reproducible source, weights and predictions audit. No training or tuning."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from engines import ROOT,UP,Tutorial,Paper,keras,tf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text_from_id(name):
    v=[dict(zip('blps',['bin','lay','place','set'])),dict(zip('bgrw',['blue','green','red','white'])),dict(zip('abiw',['at','by','in','with'])),{c:c for c in 'abcdefghijklmnopqrstuvxyz'},dict(zip('z123456789',['zero','one','two','three','four','five','six','seven','eight','nine'])),dict(zip('anps',['again','now','please','soon']))]
    return ' '.join(m[c] for m,c in zip(v,name[-6:]))
def distance(a,b):
    a=a.split();b=b.split();d=np.zeros((len(a)+1,len(b)+1),int);d[:,0]=range(len(a)+1);d[0,:]=range(len(b)+1)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
    return int(d[-1,-1])
def stats(rows,key='text'):
    return {'clips':len(rows),'errors':sum(distance(r['reference'],r[key]) for r in rows),'words':sum(len(r['reference'].split()) for r in rows),'wer':sum(distance(r['reference'],r[key]) for r in rows)/sum(len(r['reference'].split()) for r in rows),'exact':sum(r['reference']==r[key] for r in rows)}
def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def run(kind,count):
    folder=ROOT/'reports';start=time.monotonic();rows=[];failures=[]
    if kind=='tutorial':
        m=Tutorial();sources=[('s1',p,False) for p in sorted(Path('work/grid-return/data/s1').glob('*.mpg'),key=lambda p:sha_id(p.stem))[:count]]
        fidelity={'parameters':m.model.count_params(),'source_model_and_loader_executed_directly':True,'initialization_sha256':sha(ROOT/'models/tutorial-published.npz'),'local_training_performed':False,'pretraining_split_unknown':True}
        # Compare against a separately instantiated notebook model, including loss.
        from engines import notebook_context
        ctx=notebook_context();ctx['model'].set_weights(m.model.get_weights());x=m.prepare(sources[0][1]);p=m.model(x[None],training=False);q=ctx['model'](x[None],training=False)
        fidelity['separate_source_model_probability_max_error']=float(np.abs(p-q).max());assert fidelity['separate_source_model_probability_max_error']==0
        labels=np.zeros((1,40),np.int64);target=m.ctx['char_to_num'](list(text_from_id(sources[0][1].stem))).numpy();labels[0,:len(target)]=target
        loss=m.ctx['CTCLoss'](labels,p);fidelity['original_padded_target_ctc_loss']=float(tf.reduce_mean(loss));assert np.isfinite(loss).all()
    else:
        m=Paper('unseen');samples=UP/'evaluation/samples'
        sources=[('upstream-mouth',samples/'bbaf2n',True),('upstream-full-face',samples/'id2_vcd_swwp2s.mpg',False)]
        sources += [('s1',p,False) for p in sorted(Path('work/grid-return/data/s1').glob('*.mpg'),key=lambda p:sha_id(p.stem))[:count//2]]
        sources += [('s2',p,False) for p in sorted(Path('work/grid-return/data/s2').glob('*.mpg'),key=lambda p:sha_id(p.stem))[:count//2]]
        fidelity={'parameters':m.model.count_params(),'source_model2_executed_directly':True,'weights_sha256':sha(m.weights),'preprocessing':'Upstream Video.get_frames_mouth and Video.set_data; RGB /255','decoder':'Keras 2.0.2 CTC beam200 (log epsilon 1e-8) then upstream Spell','local_training_performed':False,'declared_unseen_speakers':['s1','s2','s20','s22'],'local_subset_not_full_official_benchmark':True,'gru_config':[l.forward_layer.get_config() for l in m.model.layers if isinstance(l,keras.layers.Bidirectional)]}
        import h5py
        with h5py.File(m.weights) as f:
            original=[]
            for layer in f.attrs['layer_names']:
                g=f[layer]
                for name in g.attrs.get('weight_names',[]):original.append(np.asarray(g[name]))
        actual=m.lipnet.model.get_weights();assert len(actual)==len(original)
        errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,original)];assert max(errors)==0
        fidelity['all_weight_tensor_max_errors']=errors
    for i,(speaker,path,mouth) in enumerate(sources):
        try:
            pred,p=m.predict(path,mouth=mouth) if kind=='paper' else m.predict(path)
            rows.append({'id':speaker+'/'+path.name,'reference':text_from_id(path.stem),**pred})
        except Exception as e:
            failures.append({'id':speaker+'/'+path.name,'error':str(e)})
            rows.append({'id':speaker+'/'+path.name,'reference':text_from_id(path.stem),'text':'','raw':'','error':str(e)})
        write(folder/(kind+'-progress.json'),{'completed':i+1,'total':len(sources),'failures':failures})
        print(json.dumps(rows[-1] if len(rows) and rows[-1]['id']==speaker+'/'+path.name else failures[-1]),flush=True)
    result={'engine':kind,'fidelity':fidelity,'metrics':stats(rows) if rows else None,'raw_metrics':stats(rows,'raw') if rows else None,'predictions':rows,'failures':failures,'elapsed_seconds':time.monotonic()-start,'selection_tuned_on_results':False}
    if kind=='paper':result['local_unseen_subset_metrics']=stats([r for r in rows if r['id'].split('/')[0] in ['s1','s2']]) if any(r['id'].startswith(('s1/','s2/')) for r in rows) else None
    write(folder/(kind+'-verification.json'),result);print(json.dumps(result['metrics']),flush=True)
def sha_id(s):return hashlib.sha256(('exact-lipnet-20260919:'+s).encode()).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['tutorial','paper']);p.add_argument('--count',type=int,default=100);a=p.parse_args();run(a.engine,a.count)
