"""Check historical GRU math, RGB resize, gradients and weight round trips."""
import ast,json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from engines import ROOT,UP,tf,keras,setup_paper_compat,imresize_uint8,Tutorial,Paper,paper_ctc_decode,paper_ctc_batch_cost

def main():
    setup_paper_compat()
    from keras.layers.recurrent import GRU
    rng=np.random.default_rng(41);x=rng.normal(size=(1,4,5)).astype(np.float32);g=GRU(3,return_sequences=True);g(x)
    w=[rng.normal(0,.3,size=v.shape).astype(np.float32) for v in g.get_weights()];g.set_weights(w);actual=g(x).numpy();h=np.zeros((1,3),np.float32);expected=[]
    for t in range(4):
        a=x[:,t]@w[0]+w[2];z=np.clip(.2*(a[:,:3]+h@w[1][:,:3])+.5,0,1);r=np.clip(.2*(a[:,3:6]+h@w[1][:,3:6])+.5,0,1)
        hh=np.tanh(a[:,6:]+(r*h)@w[1][:,6:]);h=z*h+(1-z)*hh;expected.append(h.copy())
    error=float(np.abs(actual-np.stack(expected,axis=1)).max());assert error<2e-7,error
    code=(ROOT/'upstream/scipy-0.19-pilutil.py').read_text().replace('.tostring()', '.tobytes()').replace('Image.isImageType(im)', 'isinstance(im, Image.Image)')
    nodes=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
    scope={n:getattr(np,n) for n in ['amin','amax','ravel','asarray','arange','ones','newaxis','transpose','iscomplexobj','uint8','issubdtype','array']};scope.update(numpy=np,Image=Image)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'scipy-0.19-pilutil.py','exec'),scope)
    pixels=rng.integers(0,256,(288,360,3),dtype=np.uint8);a=scope['imresize'](pixels,(153,191));b=imresize_uint8(pixels,(153,191));assert np.array_equal(a,b)
    report={'legacy_gru_numpy_reference_max_error':error,'legacy_gru_reset_after':g.reset_after,'scipy_rgb_resize_byte_identical':True,'checks':[],'full_training_performed':False}
    # Execute the historical backend bodies with renamed TensorFlow APIs.
    import types
    oldtf=types.SimpleNamespace(**{n:getattr(tf,n) for n in ['squeeze','transpose','expand_dims']},to_int32=lambda v:tf.cast(v,tf.int32),log=tf.math.log,sparse_to_dense=tf.compat.v1.sparse_to_dense)
    scope={'tf':oldtf,'ctc':tf.compat.v1.nn,'ctc_label_dense_to_sparse':keras.backend.ctc_label_dense_to_sparse}
    tree=ast.parse((ROOT/'upstream/keras-2.0.2-tensorflow-backend.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['ctc_decode','ctc_batch_cost']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'keras-2.0.2-backend','exec'),scope)
    prob=tf.nn.softmax(rng.normal(size=(2,9,4)).astype(np.float32));labels=np.array([[0,1],[1,0]]);length=np.array([[9],[9]]);target=np.array([[2],[2]])
    a=scope['ctc_batch_cost'](labels,prob,length,target).numpy();b=paper_ctc_batch_cost(labels,prob,length,target).numpy();assert np.array_equal(a,b)
    for greedy in [True,False]:
        a,ap=scope['ctc_decode'](prob,[9,9],greedy,200);b,bp=paper_ctc_decode(prob,[9,9],greedy,200)
        assert np.array_equal(a[0],b[0]) and np.array_equal(ap,bp)
    assert keras.backend.ctc_batch_cost is not paper_ctc_batch_cost
    report['historical_ctc_loss_and_decoders_identical']=True
    report['paper_ctc_log_epsilon']=1e-8
    for kind in ['tutorial','paper']:
        m=Tutorial() if kind=='tutorial' else Paper()
        x=m.prepare(ROOT/'samples/bras9a.mpg') if kind=='tutorial' else m.prepare(UP/'evaluation/samples/bbaf2n',mouth=True)[0]
        before=m.model(x[None],training=False).numpy()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.h5';m.model.save_weights(path)
            clone=keras.models.clone_model(m.model);clone.load_weights(path);after=clone(x[None],training=False).numpy()
            assert np.allclose(before,after,atol=1e-7);reload_error=float(np.abs(before-after).max())
        original=[v.numpy().copy() for v in m.model.trainable_weights]
        with tf.GradientTape() as tape:
            p=m.model(x[None],training=True)
            if kind=='tutorial':
                labels=np.zeros((1,40),np.int64);v=m.ctx['char_to_num'](list('bin red at s nine again')).numpy();labels[0,:len(v)]=v
                loss=tf.reduce_mean(m.ctx['CTCLoss'](labels,p))
            else:
                from lipnet.lipreading.helpers import text_to_labels
                y=text_to_labels('bin blue at f two now');labels=np.full((1,32),-1,np.float32);labels[0,:len(y)]=y
                from lipnet.core.loss import ctc_lambda_func
                loss=tf.reduce_mean(ctc_lambda_func([p,labels,np.array([[75]]),np.array([[len(y)]])]))
        grads=tape.gradient(loss,m.model.trainable_weights);assert np.isfinite(loss) and all(v is not None and np.isfinite(v.numpy()).all() for v in grads)
        opt=keras.optimizers.legacy.Adam(learning_rate=1e-4,epsilon=1e-7 if kind=='tutorial' else 1e-8);opt.apply_gradients(zip(grads,m.model.trainable_weights))
        changed=sum(not np.array_equal(a,v.numpy()) for a,v in zip(original,m.model.trainable_weights));assert changed
        report['checks'].append({'engine':kind,'reload_probability_max_error':reload_error,'finite_ctc_loss':float(loss),'finite_gradients':True,'changed_trainable_tensors':changed,'note':'One disposable single-example optimizer smoke step, not a training run. Published weight files unchanged.'})
    (ROOT/'reports/compatibility-checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
