"""Reconstruct identical LRS2 media from pinned, SHA-verified original shards."""
import hashlib,json,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'work/corpora/lrs2'
rows=json.loads((DATA/'manifest.json').read_text())
expected={r['id']:r for r in rows}
sources=json.loads((DATA/'provenance.json').read_text())['download_sources']
CACHE=ROOT/'work/lightning/shards'; CACHE.mkdir(parents=True,exist_ok=True)
(DATA/'videos').mkdir(exist_ok=True)
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def one(s):
 p=CACHE/(s['split']+'-'+Path(s['file']).name)
 if not p.exists() or digest(p)!=s['sha256']:
  for attempt in range(4):
   try:
    with requests.get(s['url'],stream=True,timeout=(30,120)) as response:
     response.raise_for_status()
     with p.with_suffix('.partial').open('wb') as f:
      for chunk in response.iter_content(4*1024*1024):f.write(chunk)
    assert digest(p.with_suffix('.partial'))==s['sha256'],'Shard hash mismatch'
    p.with_suffix('.partial').replace(p);break
   except Exception:
    if attempt==3:raise
    time.sleep(2**attempt)
 seen=[]
 for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
  for item in batch.to_pylist():
   r=expected[item['sample_id']]
   assert r['split']==s['split'] and r['source_revision']==s['revision']
   assert r['declared_frames']==item['length']
   from training.metrics import normalize
   assert normalize(item['label'])==r['text']
   raw=item['video']; assert isinstance(raw,bytes)
   assert hashlib.sha256(raw).hexdigest()==r['sha256']
   target=DATA/'videos'/Path(r['path']).name
   if not target.exists() or digest(target)!=r['sha256']:
    temporary=target.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(target)
   seen.append(r['id'])
 p.unlink() # Verified video bytes are retained; no redundant 11GB shard cache.
 print(json.dumps(dict(shard=s['file'],split=s['split'],clips=len(seen))),flush=True)
 return seen
all_ids=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(one,s) for s in sources]):all_ids.extend(future.result())
assert len(all_ids)==len(set(all_ids))==len(rows) and set(all_ids)==set(expected)
report=dict(verified_clips=len(rows),source_shards=len(sources),all_video_hashes_match=True,
            all_ids_transcripts_lengths_splits_match=True,completed_unix=time.time())
(ROOT/'work/lightning/data-transfer-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
