"""Source-backed Keras runtimes. No custom alignment, grammar beam or fine-tuning."""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import ast,json,sys,types,threading,time
import numpy as np
import tensorflow as tf
import tf_keras as keras
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parent
UP=ROOT/'upstream/rizkiarm'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

def notebook_context():
    nb=json.loads((ROOT/'upstream/LipNet.ipynb').read_text())
    ctx={'tf':tf,'cv2':cv2,'np':np,'os':os,'List':list,'Sequential':keras.Sequential,**{n:getattr(keras.layers,n) for n in ['Conv3D','Activation','MaxPool3D','TimeDistributed','Flatten','Bidirectional','LSTM','Dropout','Dense']}}
    vocab=list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
    ctx['char_to_num']=keras.layers.StringLookup(vocabulary=vocab,oov_token='')
    ctx['num_to_char']=keras.layers.StringLookup(vocabulary=ctx['char_to_num'].get_vocabulary(),oov_token='',invert=True)
    for cell in nb['cells']:
        code=''.join(cell.get('source',[]))
        if cell['cell_type']!='code':continue
        if code.startswith('model = Sequential()'):
            exec(compile(code,'upstream/LipNet.ipynb:model','exec'),ctx)
        else:
            try:tree=ast.parse(code)
            except SyntaxError:continue
            defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load_video','load_alignments','load_data','CTCLoss','scheduler']]
            if defs:exec(compile(ast.Module(body=defs,type_ignores=[]),'upstream/LipNet.ipynb:functions','exec'),ctx)
    return ctx

class Tutorial:
    def __init__(self):
        self.ctx=notebook_context();self.model=self.ctx['model'];self.lock=threading.Lock()
        with np.load(ROOT/'models/tutorial-published.npz') as z:self.model.set_weights([z[f'w{i}'] for i in range(20)])
        assert self.model.count_params()==8471924
    def prepare(self,path):
        x=self.ctx['load_video'](str(path))
        if len(x)>75 or tuple(x.shape[1:])!=(46,140,1):raise ValueError('Use original 25-fps GRID video in tutorial mode.')
        return tf.pad(x,[[0,75-len(x)],[0,0],[0,0],[0,0]]).numpy()
    def predict(self,path):
        start=time.perf_counter();x=self.prepare(path)
        with self.lock:p=self.model(x[None],training=False).numpy()
        decoded=keras.backend.ctc_decode(p,[75],greedy=True)[0][0]
        text=tf.strings.reduce_join(self.ctx['num_to_char'](decoded[0])).numpy().decode()
        return {'text':text,'raw':text,'seconds':time.perf_counter()-start,'mode':'Original notebook fixed crop, grayscale arithmetic and greedy CTC','input_shape':list(x.shape)},p

# These are the old SciPy functions used by videos.py, limited to its uint8 RGB path.
def imresize_uint8(arr,size,interp='bilinear',mode=None):
    assert np.asarray(arr).dtype==np.uint8 and np.asarray(arr).ndim==3
    return np.asarray(Image.fromarray(arr,mode=mode).resize((size[1],size[0]),resample=Image.Resampling.BILINEAR))

def paper_ctc_batch_cost(y_true,y_pred,input_length,label_length):
    # Keras 2.0.2 used literal 1e-8, not the modern backend epsilon (1e-7).
    # Keep lengths rank one for batch size one (old squeeze was overly broad).
    lengths=tf.cast(tf.reshape(label_length,[-1]),tf.int32)
    inputs=tf.cast(tf.reshape(input_length,[-1]),tf.int32)
    labels=tf.cast(keras.backend.ctc_label_dense_to_sparse(y_true,lengths),tf.int32)
    logits=tf.math.log(tf.transpose(y_pred,[1,0,2])+1e-8)
    return tf.expand_dims(tf.compat.v1.nn.ctc_loss(labels=labels,inputs=logits,sequence_length=inputs),1)

def paper_ctc_decode(y_pred,input_length,greedy=True,beam_width=100,top_paths=1):
    logits=tf.math.log(tf.transpose(y_pred,[1,0,2])+1e-8)
    lengths=tf.cast(input_length,tf.int32)
    if greedy:
        decoded,prob=tf.nn.ctc_greedy_decoder(logits,lengths)
    else:
        decoded,prob=tf.compat.v1.nn.ctc_beam_search_decoder(logits,lengths,beam_width=beam_width,top_paths=top_paths)
    return [tf.sparse.to_dense(v,default_value=-1) for v in decoded],prob

def setup_paper_compat():
    if str(UP) not in sys.path:sys.path.insert(0,str(UP))
    # The source classes/methods remain unchanged; legacy import locations moved.
    import scipy.misc,scipy.ndimage,imageio.v2 as imageio
    scipy.misc.imresize=imresize_uint8;scipy.ndimage.imread=imageio.imread
    # sk-video 1.1.11 uses NumPy aliases removed in NumPy 2.
    np.float=float;np.int=int
    # A proxy keeps paper-era CTC defaults out of the notebook's TF2 backend.
    backend=types.ModuleType('keras.backend');backend.__dict__.update(vars(keras.backend))
    backend.ctc_batch_cost=paper_ctc_batch_cost;backend.ctc_decode=paper_ctc_decode
    package=types.ModuleType('keras');package.__dict__.update(vars(keras));package.backend=backend
    sys.modules['keras']=package
    for name,module in [('keras.backend',backend),('keras.models',keras.models),('keras.layers',keras.layers)]:sys.modules[name]=module
    for name in ['core','convolutional','pooling','wrappers','normalization']:sys.modules['keras.layers.'+name]=keras.layers
    legacy=types.ModuleType('keras.layers.recurrent')
    # Keras 2.0.2 GRU: hard sigmoid gates, reset BEFORE multiplication.
    class LegacyGRU(keras.layers.GRU):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('recurrent_activation','hard_sigmoid');kwargs.setdefault('reset_after',False)
            super().__init__(*args,**kwargs)
    keras.utils.get_custom_objects()['LegacyGRU']=LegacyGRU
    legacy.GRU=LegacyGRU;sys.modules['keras.layers.recurrent']=legacy

class Paper:
    def __init__(self,scenario='unseen'):
        setup_paper_compat()
        from lipnet.model2 import LipNet
        from lipnet.lipreading.videos import Video
        from lipnet.utils.spell import Spell
        from lipnet.lipreading.helpers import labels_to_text
        self.Video=Video;self.labels_to_text=labels_to_text;self.scenario=scenario;self.lock=threading.Lock()
        self.lipnet=LipNet(frames_n=None);self.model=keras.Model(self.lipnet.input_data,self.lipnet.y_pred)
        self.weights=UP/'evaluation/models'/({'unseen':'unseen-weights178.h5','overlapped':'overlapped-weights368.h5'}[scenario])
        self.lipnet.model.load_weights(str(self.weights))
        self.spell=Spell(str(UP/'common/dictionaries/grid.txt'))
    def prepare(self,path,mouth=False):
        v=self.Video(vtype='mouth' if mouth else 'face',face_predictor_path=str(UP/'common/predictors/shape_predictor_68_face_landmarks.dat'))
        if Path(path).is_dir():v.from_frames(str(path))
        else:v.from_video(str(path))
        if v.data.ndim!=4 or v.data.shape[1:]!=(100,50,3):raise ValueError(f'Original mouth detector could not prepare this video: {v.data.shape}. Keep one face fully visible.')
        return np.array([v.data]).astype(np.float32)/255
    def predict(self,path,mouth=False):
        start=time.perf_counter();x=self.prepare(path,mouth)
        return self._predict_input(x,start)
    def predict_frames(self,frames):
        start=time.perf_counter()
        v=self.Video(vtype='face',face_predictor_path=str(UP/'common/predictors/shape_predictor_68_face_landmarks.dat')).from_array(frames)
        if v.data.ndim!=4 or v.data.shape[1:]!=(100,50,3):raise ValueError('Original mouth detector could not track the face. Try again in good light.')
        return self._predict_input(np.array([v.data]).astype(np.float32)/255,start)
    def _predict_input(self,x,start):
        with self.lock:p=self.model(x,training=False).numpy()
        paths=paper_ctc_decode(p,[x.shape[1]],greedy=False,beam_width=200,top_paths=1)[0][0].numpy()
        raw=self.labels_to_text(paths[0]);text=self.spell.sentence(raw)
        return {'text':text,'raw':raw,'seconds':time.perf_counter()-start,'mode':f'Original model2 {self.scenario} checkpoint; dlib crop; beam 200 and upstream spelling','input_shape':list(x.shape[1:])},p
