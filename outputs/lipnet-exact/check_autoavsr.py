from pathlib import Path
import hashlib,json,time,cv2,numpy as np
from autoavsr_bridge import AutoAVSR,ROOT
from live import LiveSession
start=time.perf_counter();reader=AutoAVSR();load=time.perf_counter()-start
source=ROOT.parent/'lipreading-studio/examples/research-demo.mp4'
upload,_=reader.predict(source)
cap=cv2.VideoCapture(str(source));frames=[]
while len(frames)<200:
 ok,frame=cap.read()
 if not ok:break
 h,w=frame.shape[:2];scale=min(1,640/max(h,w));frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
 frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
cap.release()
frames=np.stack(frames)
try:
 reader.predict_frames(np.zeros((10,641,10,3),dtype=np.uint8))
 raise AssertionError('Oversize frames must fail before pipe write')
except ValueError:pass
recorded,_=reader.predict_frames(frames)
session=LiveSession(lambda x:json.dumps(reader.predict_frames(x)[0]),seconds=8)
for i,frame in enumerate(frames):session.accept(frame,i/25)
session.on_ended();assert session.wait_until_idle(timeout=180)
state=session.snapshot();assert len(state['segments'])==1 and not state['segments'][0].get('error'),state
worker=json.loads(state['segments'][0]['text']);assert worker['text']==recorded['text']
assert upload['checkpoint']==recorded['checkpoint']=='autoavsr' and upload['text'] and recorded['text']
report=dict(model_load_seconds=load,upload=upload,recorded_frames=recorded,recorded_webcam_worker=worker,recorded_worker_text_matches=True,physical_camera_tested=False,lipnet_preserved=True)
(ROOT/'reports/autoavsr-integration.json').write_text(json.dumps(report,indent=2)+'\n')
reader.close()
selection=json.loads((ROOT/'models/paper-selection.json').read_text())
author=next((ROOT/'upstream/rizkiarm').rglob('unseen-weights178.h5'))
assert hashlib.sha256(author.read_bytes()).hexdigest()==selection['author_sha256']
assert hashlib.sha256((ROOT/selection['candidate_path']).read_bytes()).hexdigest()==selection['candidate_sha256']
report.update(lipnet_author_sha256=selection['author_sha256'],lipnet_candidate_sha256=selection['candidate_sha256'],oversized_frames_rejected_before_send=True)
(ROOT/'reports/autoavsr-integration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
