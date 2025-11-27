from pathlib import Path
import concurrent.futures
import hashlib
import json
import requests
import time

ROOT = Path(__file__).resolve().parent / 'data'
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
 'train': ('MahmoodAnaam/LRS2-Train', '26dea093e50d1ee89fe717648b50922cc4b4011b', ['data/train-00000-of-00022.parquet', 'data/train-00011-of-00022.parquet']),
 'validation': ('MahmoodAnaam/LRS2-Validation', '5f987b9729b85720a81b2b521491cc204c19ed6c', ['data/validation-00000-of-00001.parquet']),
 'test': ('MahmoodAnaam/LRS2-Test', 'e14bf18868ca78bd97b0b9ce6615b262a32dfdf1', ['data/test-00000-of-00001.parquet'])
}

def download(item):
 split, repo, revision, file = item
 dest=ROOT / Path(file).name
 url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}'
 if not dest.exists():
  part=dest.with_suffix('.part')
  for attempt in range(4):
   try:
    with requests.get(url, stream=True, timeout=(30,120)) as r:
     r.raise_for_status()
     count=0
     with part.open('wb') as f:
      for chunk in r.iter_content(1024*1024):
       count+=len(chunk)
       if count>800_000_000: raise ValueError('File exceeds bounded download limit')
       f.write(chunk)
    part.replace(dest)
    break
   except requests.RequestException:
    if attempt==3:raise
    time.sleep(2*(attempt+1))
 digest=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
 row={'split':split,'repository':repo,'revision':revision,'file':file,'local_file':str(dest),'bytes':dest.stat().st_size,'sha256':digest,'url':url}
 print(json.dumps(row),flush=True)
 return row

if __name__=='__main__':
 items=[(split,repo,rev,f) for split,(repo,rev,files) in SOURCES.items() for f in files]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  rows=list(pool.map(download,items))
 (ROOT/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n')
