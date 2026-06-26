import time,json,hashlib
from pathlib import Path
import cv2
root=Path('/teamspace/studios/this_studio/lipreader')
rows=json.loads((root/'work/corpora/lrs2/manifest.json').read_text())[:40]
result=[]
for nt in (0,1):
 tick=time.monotonic(); hashes=[]
 for r in rows:
  path=str(root/'work/corpora/lrs2/videos'/Path(r['path']).name)
  cap=cv2.VideoCapture(path) if nt==0 else cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,nt])
  h=hashlib.sha256();n=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   h.update(frame.tobytes());n+=1
  cap.release(); assert n==r['declared_frames'];hashes.append(h.hexdigest())
 result.append(dict(threads=nt,seconds=time.monotonic()-tick,hashes=hashes))
assert result[0]['hashes']==result[1]['hashes']
print(json.dumps([dict(threads=x['threads'],seconds=x['seconds']) for x in result]),flush=True)
