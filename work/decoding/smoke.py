import sys,json,time,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch,numpy as np
from studio.pipeline import LipReader,ROOT
from studio.language_model import load_language_model
reader=LipReader();lm=load_language_model(ROOT/'models/english-lm',reader.tokens)
torch.manual_seed(19);tokens=torch.randint(1,len(reader.tokens)-1,(3,8));tokens[:,0]=len(reader.tokens)-1
max_error=0.;states=[None]*3
with torch.inference_mode():
 for length in range(1,9):
  prefix=tokens[:,:length]
  scores,states=lm.batch_score(prefix,states,torch.zeros(3,1,1))
  expected=lm.logits(prefix)[0][:,-1].log_softmax(-1)
  torch.testing.assert_close(scores,expected,atol=2e-5,rtol=1e-5)
  max_error=max(max_error,float((scores-expected).abs().max()))
 # Compare our layer equations directly with the authors' layer implementation.
 spec=importlib.util.spec_from_file_location('author_lm_layer','work/decoding/transformer_encoder_layer.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
 from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
 ref=mod.EncoderLayer(512,MultiHeadedAttention(8,512,0.),PositionwiseFeedForward(512,2048,0.),None,0.).eval()
 ref.load_state_dict(lm.encoder.encoders[0].state_dict())
 features=torch.randn(2,7,512);mask=torch.ones(2,7,7,dtype=torch.bool).tril()
 expected=ref(features,mask)[0];actual=lm.encoder.encoders[0](features,mask)
 torch.testing.assert_close(actual,expected,atol=0,rtol=0)
print('language-model parity passed',flush=True)
rows=json.loads(Path('work/overnight/dataset/manifest.json').read_text());rows=[r for r in rows if r['split']=='validation'][:4]
from training.experiment import Experiment
holder=type('FeatureHolder',(),{'cache':Path('work/overnight/features'),'reader':reader})()
results=[]
for beam,weight in [(5,0.),(20,0.),(20,.3),(40,.3)]:
 reader.configure_decoding(beam,ROOT/'models/english-lm' if weight else None,weight)
 for row in rows:
  encoded=Experiment.features(holder,row);tick=time.monotonic();prediction=reader.decode(encoded)
  results.append({'beam':beam,'lm_weight':weight,'id':row['id'],'reference':row['text'],'prediction':prediction,'seconds':time.monotonic()-tick})
 print(beam,weight,[(r['prediction'],r['seconds']) for r in results[-4:]],flush=True)
report={'lm_parameters':sum(p.numel() for p in lm.parameters()),'incremental_full_prefix_max_abs_error':max_error,'author_layer_exact_parity':True,'smoke_predictions':results,'not_an_accuracy_benchmark':True}
(ROOT/'validation/decoding-feasibility.json').write_text(json.dumps(report,indent=2)+'\n')
p=ROOT/'models/english-lm/provenance.json';d=json.loads(p.read_text());d['parameters']=report['lm_parameters'];p.write_text(json.dumps(d,indent=2)+'\n')
