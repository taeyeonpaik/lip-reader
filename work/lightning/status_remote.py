import json
from pathlib import Path
p=Path('/teamspace/studios/this_studio/lipreader')
f=p/'work/lightning/data-restore.log'
records=[]
if f.exists():
 for line in f.read_text().splitlines():
  try: records.append(json.loads(line))
  except ValueError:pass
out={'restored_clips':sum(r.get('clips',0) for r in records),'total_clips':48139,'shards_finished':sum('shard' in r for r in records)}
for name,rel in [('transfer','work/lightning/data-transfer-verification.json'),('progress','outputs/lipreading-studio/training/runs/lrs2-cuda-v1/progress.json'),('readiness','outputs/lipreading-studio/training/runs/lrs2-cuda-v1/cpu-readiness.json')]:
 f=p/rel
 if f.exists():out[name]=json.loads(f.read_text())
f=p/'outputs/lipreading-studio/training/runs/lrs2-batched-v1/progress.json'
if f.exists():out['batched_progress']=json.loads(f.read_text())
print(json.dumps(out,indent=2))
