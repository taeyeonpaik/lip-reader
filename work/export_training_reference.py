import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'outputs/lipnet-tutorial'))
from tutorial.core import build_model,load_model,read_video,encode,ctc_loss
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
dest=root/'work/gpu-training-reference';dest.mkdir(exist_ok=True)
keras.utils.set_random_seed(42)
initial=build_model()
np.savez(dest/'initial.npz',**{f'w{i}':w for i,w in enumerate(initial.get_weights())})
model=load_model(root/'outputs/lipnet-tutorial/models/tutorial.weights.h5')
np.savez(dest/'published.npz',**{f'w{i}':w for i,w in enumerate(model.get_weights())})
frames,_=read_video(root/'outputs/lipnet-tutorial/samples/bras9a/bras9a.mpg')
labels=encode('bin red at s nine again')
probabilities=model(frames[None],training=False).numpy()
np.savez(dest/'reference.npz',frames=frames,probabilities=probabilities,labels=labels,
         loss=ctc_loss(labels[None],probabilities).numpy())
print('Exported initial random weights and published checkpoint parity reference')
