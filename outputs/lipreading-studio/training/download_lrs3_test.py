from pathlib import Path
import requests,json,hashlib,time
meta=Path(__file__).resolve().parent/'runs/lrs3-official-test-v1'
root=Path('/private/tmp/lrs3-official-test');root.mkdir(exist_ok=True)
api=json.loads((meta/'source-api.json').read_text());files=json.loads((meta/'source-files.json').read_text())
assert api['gated'] is False and not api['private']
for record in files:
 if not record['path'].endswith('.parquet'):continue
 target=root/Path(record['path']).name
 if target.exists():
  with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  if sha==record['lfs']['oid']:print('verified cache',target.name,flush=True);continue
 url=f"https://huggingface.co/datasets/{api['id']}/resolve/{api['sha']}/{record['path']}"
 temp=target.with_suffix('.part');h=hashlib.sha256();size=0
 with requests.get(url,stream=True,timeout=(20,120)) as response:
  response.raise_for_status()
  with temp.open('wb') as out:
   for block in response.iter_content(4*1024*1024):
    out.write(block);h.update(block);size+=len(block)
 assert size==record['size'] and h.hexdigest()==record['lfs']['oid'],target.name
 temp.replace(target);print('download verified',target.name,size,flush=True)
print('COMPLETE',flush=True)
