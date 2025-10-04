import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys,json
from pathlib import Path
from typing import List
sys.path.insert(0,str(Path('outputs/lipnet-tutorial').resolve()))
import cv2,numpy as np,tensorflow as tf,tf_keras as keras
from tutorial.core import build_model,read_video,encode,tutorial_ctc_loss,decode
source=json.loads(Path('work/tutorial-reference/LipNet.ipynb').read_text())
cells=[''.join(c.get('source',[])) for c in source['cells']]
ctx={'tf':tf,'cv2':cv2,'List':List,'Sequential':keras.Sequential,
     **{k:getattr(keras.layers,k) for k in ['Conv3D','Activation','MaxPool3D','TimeDistributed','Flatten','Bidirectional','LSTM','Dropout','Dense']}}
ctx['char_to_num']=keras.layers.StringLookup(vocabulary=list("abcdefghijklmnopqrstuvwxyz'?!123456789 "),oov_token='')
exec(cells[9],ctx);exec(cells[38],ctx)
reference=ctx['model'];ours=build_model();ours.load_weights('outputs/lipnet-tutorial/models/tutorial.weights.h5')
reference.set_weights(ours.get_weights())
rows=json.loads(Path('outputs/lipnet-tutorial/validation/ten-clip-evaluation.json').read_text())['predictions'][:2]
inputs=[];errors=[]
for row in rows:
 path=row['video'];ref=ctx['load_video'](path).numpy();got,_=read_video(path)
 errors.append(float(np.max(np.abs(ref-got))));inputs.append(got)
x=np.stack(inputs);r=reference(x,training=False);p=ours(x,training=False)
# The original loss body uses tf.keras, so provide its contemporary tf_keras backend explicitly.
loss_source=cells[48].replace('tf.keras.backend.ctc_batch_cost','keras.backend.ctc_batch_cost')
ctx['keras']=keras;exec(loss_source,ctx)
labels=np.zeros((2,40),np.int64)
for i,row in enumerate(rows):
 y=encode(row['reference']);labels[i,:len(y)]=y
loss_ref=ctx['CTCLoss'](labels,p);loss_ours=tutorial_ctc_loss(labels,p)
report={'reference':'nicknochnack/LipNet LipNet.ipynb cells 9, 38, 48',
 'parameters':ours.count_params(),'reference_parameters':reference.count_params(),
 'input_maximum_absolute_errors':errors,'probability_maximum_absolute_error':float(tf.reduce_max(tf.abs(r-p))),
 'tutorial_ctc_maximum_absolute_error':float(tf.reduce_max(tf.abs(loss_ref-loss_ours))),
 'predictions':decode(p)[0],'references':[row['reference'] for row in rows],
 'checkpoint':'Authors published 96-epoch checkpoint, not locally trained',
 'local_training_completed_epochs':2,
 'recent_200_epoch_run':'Different paper-style GRU model; not this tutorial model',
 'differences':['Fixed seed and frozen file split for reproducibility','tf_keras compatibility runtime instead of TensorFlow 2.10.1','Input validation and 74-frame padding moved into the loader','Additional optional corrected-length loss exists; the two-epoch run used the original padded-target loss']}
assert report['parameters']==report['reference_parameters']==8471924
assert max(errors)==0 and report['probability_maximum_absolute_error']<1e-6 and report['tutorial_ctc_maximum_absolute_error']<1e-5
Path('outputs/lipnet-tutorial/validation/tutorial-fidelity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
