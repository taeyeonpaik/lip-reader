from pathlib import Path
import sys,json,shutil,os,gc
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import torch
from training import lrs2_finetune as t
from training.data import atomic_json
rows=t.prepare();e=t.Experiment(rows)
old=json.loads((t.RUN/'config-before-speedup.json').read_text())
allowed={'source_sha256','execution_training_cpu_threads','execution_training_decoder_device','execution_evaluation_cpu_threads','decoded_verification_proof_sha256'}
assert all(e.config[k]==v for k,v in old.items() if k not in allowed)
last=t.STORAGE/'last.pt';backup=t.STORAGE/'before-speedup.pt'
assert not backup.exists()
saved=torch.load(last,map_location='cpu',weights_only=True,mmap=True)
assert saved['config']==old and saved['progress']['updates']==100
original_sha=t.sha(last);original_progress=dict(saved['progress'])
# Moving the file keeps the pinned mmap alive and preserves the untouched backup.
os.rename(last,backup)
saved['config']=e.config
try:
 t.save_torch(saved,last)
except Exception:
 if not last.exists():os.rename(backup,last)
 raise
fresh=torch.load(last,map_location='cpu',weights_only=True,mmap=True)
assert fresh['progress']==original_progress and fresh['config']==e.config
assert all(torch.equal(v,fresh['model'][k]) for k,v in saved['model'].items())
for key,state in saved['optimizer']['state'].items():
 for name,value in state.items():
  other=fresh['optimizer']['state'][key][name]
  assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
assert saved['optimizer']['param_groups']==fresh['optimizer']['param_groups']
assert t.sha(backup)==original_sha
atomic_json(t.RUN/'config.json',e.config)
atomic_json(t.RUN/'execution-optimization.json',dict(state='verified',original_checkpoint_sha256=original_sha,backup=str(backup),new_checkpoint_sha256=t.sha(last),source_before=old['source_sha256'],source_after=e.config['source_sha256'],weights_identical=True,optimizer_identical=True,progress_identical=True,resume_updates=100,resume_offset=original_progress['offset'],unchanged_recipe_keys=[k for k in old if k not in allowed],training_cpu_threads=e.config['execution_training_cpu_threads'],training_decoder_device=e.config['execution_training_decoder_device'],evaluation_cpu_threads=4,verification='All video source hashes rechecked; reuse completed decode/pixel checks only for unchanged manifest/source bytes.'))
print('Checkpoint migration verified; weights, optimizer and progress identical.',flush=True)
