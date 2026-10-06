"""Reproducible prepared-crop inference benchmark; no training or model selection."""
import hashlib,json,platform,statistics,time,os,sys
from pathlib import Path
ROOT=Path('/teamspace/studios/this_studio/lipreader/outputs/lipreading-studio')
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from studio.pipeline import LipReader
run=ROOT/'training/runs/lrs2-batched-v1'
out=run/'latency-cpu-v1';out.mkdir(exist_ok=True)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,data):
 p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
rows=[r for r in json.loads((run/'manifest.json').read_text()) if r['split']=='test']
rows.sort(key=lambda r:(abs(r['declared_frames']-75),hashlib.sha256(r['id'].encode()).hexdigest()))
selected=rows[:20];warmup=rows[20:22]
selection=json.loads((run/'selection.json').read_text())
write('protocol.json',dict(checkpoint=selection,clip_ids=[r['id'] for r in selected],warmup_ids=[r['id'] for r in warmup],selection='20 test clips nearest 75 frames, SHA256 ID breaks ties; chosen without timing or predictions',scope='Existing mouth-crop MP4 read/decode + center crop/normalization + encoder + beam40 decoder. Excludes face tracking, upload, model initialization.',hardware=platform.platform(),cpu_threads=2,affinity_cpus=len(os.sched_getaffinity(0)),torch=str(torch.__version__),beam=40,background_load='TCD-TIMIT download/verification active on same Studio'))
print('Checking selected weights and media',flush=True)
assert sha(selection['checkpoint'])==selection['sha256']
for r in selected+warmup:assert sha(r['path'])==r['sha256']
start=time.perf_counter();reader=LipReader(device='cpu',beam_size=40)
reader.model.load_state_dict(torch.load(selection['checkpoint'],map_location='cpu',weights_only=True),strict=True)
reader.model.eval();torch.set_num_threads(2);cv2.setNumThreads(1)
load_seconds=time.perf_counter()-start
print(f'Model loaded in {load_seconds:.1f}s',flush=True)
def transcribe(row):
 start=time.perf_counter();cap=cv2.VideoCapture(row['path']);frames=[]
 try:
  fps=cap.get(cv2.CAP_PROP_FPS)
  while True:
   ok,frame=cap.read()
   if not ok:break
   frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 finally:cap.release()
 assert abs(fps-25)<.01 and len(frames)==row['declared_frames']
 crops=np.stack(frames);text=reader.recognize(crops)
 return dict(id=row['id'],frames=len(frames),fps=fps,clip_seconds=len(frames)/fps,latency_seconds=time.perf_counter()-start,prediction=text)
for row in warmup:transcribe(row)
results=[]
for row in selected:
 results.append(transcribe(row));write('progress.json',dict(completed=len(results),total=20,latest=results[-1]));print(f"{len(results)}/20: {results[-1]['clip_seconds']:.2f}s clip in {results[-1]['latency_seconds']:.2f}s",flush=True)
write('predictions.json',results)
a=[r['latency_seconds'] for r in results];d=[r['clip_seconds'] for r in results]
summary=dict(state='complete',clips=20,median_clip_seconds=statistics.median(d),clip_range_seconds=[min(d),max(d)],median_latency_seconds=statistics.median(a),p95_latency_seconds=float(np.percentile(a,95)),median_real_time_factor=statistics.median([r['latency_seconds']/r['clip_seconds'] for r in results]),model_load_seconds=load_seconds,hardware='Intel Xeon CPU @ 2.60GHz; 4 allocated vCPUs; 2 PyTorch inference threads',checkpoint_sha256=selection['sha256'],checkpoint_epoch=selection['epoch'],beam_size=40,scope='prepared LRS2 mouth crops; excludes face tracking and upload; warm model',background_load='TCD-TIMIT download/verification active; timing may be affected')
write('result.json',summary);print(json.dumps(summary),flush=True)
