"""Explicit training entry points preserving each source's model and loss.

Training is never started by the app. The caller must supply epochs and output.
"""
import argparse,json
from pathlib import Path
import numpy as np
from engines import ROOT,notebook_context,setup_paper_compat,keras,tf

class PaperData(keras.utils.Sequence):
    def __init__(self,root,split,batch):
        from lipnet.lipreading.videos import Video
        from lipnet.lipreading.aligns import Align
        from lipnet.lipreading.helpers import text_to_labels
        from lipnet.lipreading.curriculums import Curriculum
        self.Video=Video;self.Align=Align;self.encode=text_to_labels;self.root=Path(root);self.split=split;self.batch=batch
        self.paths=sorted(p for p in (self.root/split).glob('*/*') if p.is_dir())
        if not self.paths:raise ValueError('Paper data requires train|val/speaker/clip/*.png and align/clip.align, as upstream.')
        self.curriculum=Curriculum(lambda epoch:{'sentence_length':-1,'flip_probability':.5,'jitter_probability':.05});self.epoch=0
        self.on_epoch_end()
    def __len__(self):return len(self.paths)//self.batch
    def on_epoch_end(self):
        self.curriculum.update(self.epoch,train=self.split=='train');self.epoch+=1
        if self.split=='train':np.random.shuffle(self.paths)
    def __getitem__(self,index):
        x=[];ys=[];lengths=[]
        for p in self.paths[index*self.batch:(index+1)*self.batch]:
            v=self.Video().from_frames(str(p));a=self.Align(32,self.encode).from_file(self.root/'align'/(p.name+'.align'))
            v,a,_=self.curriculum.apply(v,a)
            if v.data.shape!=(75,100,50,3):raise ValueError('Invalid source mouth-frame dimensions: '+str(p))
            x.append(v.data);ys.append(a.padded_label);lengths.append(a.label_length)
        return {'the_input':np.asarray(x,np.float32)/255,'the_labels':np.asarray(ys),'input_length':np.full((len(x),1),75,np.int64),'label_length':np.asarray(lengths,np.int64)[:,None]},np.zeros(len(x))

def main():
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['tutorial','paper']);p.add_argument('--data',required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--initialization',choices=['random','published'],default='random');a=p.parse_args()
    if a.epochs<1:raise ValueError('epochs must be positive')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory; preserve prior runs.')
    out.mkdir(parents=True,exist_ok=True)
    np.random.seed(55)
    callbacks=[keras.callbacks.CSVLogger(str(out/'history.csv')),keras.callbacks.ModelCheckpoint(str(out/'epoch-{epoch:03d}.weights.h5'),save_weights_only=True)]
    if a.engine=='tutorial':
        ctx=notebook_context();model=ctx['model']
        if a.initialization=='published':
            with np.load(ROOT/'models/tutorial-published.npz') as z:model.set_weights([z[f'w{i}'] for i in range(20)])
        base=Path(a.data)
        def loader(path):
            path=Path(path.numpy().decode());return ctx['load_video'](str(path)),ctx['load_alignments'](str(base/'alignments/s1'/(path.stem+'.align')))
        def mapped(path):
            x,y=tf.py_function(loader,[path],(tf.float32,tf.int64));x.set_shape([None,46,140,1]);y.set_shape([None]);return x,y
        data=tf.data.Dataset.list_files(str(base/'s1/*.mpg'));data=data.shuffle(500,reshuffle_each_iteration=False).map(mapped).padded_batch(2,padded_shapes=([75,None,None,None],[40])).prefetch(tf.data.AUTOTUNE)
        train=data.take(450);val=data.skip(450)
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=.0001),loss=ctx['CTCLoss'])
        callbacks.append(keras.callbacks.LearningRateScheduler(lambda epoch,lr:float(ctx['scheduler'](epoch,lr))))
        recipe={'batch':2,'loss':'original padded 40-target CTCLoss','lr':.0001,'split':'original notebook take450/skip450 batches; original shuffle semantics preserved; not a new-speaker benchmark'}
    else:
        setup_paper_compat();from lipnet.model2 import LipNet
        net=LipNet();model=net.model
        if a.initialization=='published':model.load_weights(str(ROOT/'upstream/rizkiarm/evaluation/models/unseen-weights178.h5'))
        train=PaperData(a.data,'train',50);val=PaperData(a.data,'val',50)
        if min(len(train),len(val))<1:raise ValueError('Original batch50 needs at least 50 clips in each split.')
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=.0001,beta_1=.9,beta_2=.999,epsilon=1e-8),loss={'ctc':lambda actual,predicted:predicted})
        recipe={'batch':50,'lr':.0001,'epsilon':1e-8,'loss':'upstream CTC actual lengths','augmentation':'upstream Curriculum flip0.5 jitter0.05','split':'caller-supplied original repository train/val mouth frames'}
    (out/'recipe.json').write_text(json.dumps({**vars(a),**recipe,'runtime_compatibility_changes':'see COMPATIBILITY.md','full_paper_training_not_previously_run_here':True},indent=2))
    history=model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks,verbose=1)
    (out/'complete.json').write_text(json.dumps({'epochs_completed':len(history.history['loss']),'history':history.history},indent=2))
if __name__=='__main__':main()
