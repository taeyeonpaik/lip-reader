import json,time,tempfile
from pathlib import Path
import numpy as np
from tutorial.runtime import GridReader,read_rgb,fixed_input,prepare_upload,ROOT
from tutorial.live import OneTakeSession
r=GridReader();source=ROOT/'samples/bras9a/bras9a.mpg';frames=read_rgb(source)
report={'physical_camera_tested':False,'device':r.device}
ref=np.load('work/gpu-training-reference/reference.npz')
print('reference keys',ref.files,flush=True)
report['fixed_example']=r.recognize(frames,fixed=True)
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'silent.mp4';start=time.perf_counter();prepare_upload(source,path)
 report['upload']=r.recognize(read_rgb(path));report['upload_total_seconds']=time.perf_counter()-start
live=OneTakeSession(lambda f:json.dumps(r.recognize(f)))
for i,f in enumerate(frames):live.accept(f,i/25)
live.on_ended();assert live.wait_until_idle(60)
s=live.snapshot();assert len(s['segments'])==1 and not s['segments'][0].get('error'),s
report['recorded_webcam']=s
Path('work/grid-return/runtime-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
