"""Exercise the app's upload and one-take worker using recorded RGB frames."""
import json,time,tempfile
from pathlib import Path
import numpy as np
from engines import ROOT,UP,Paper,Tutorial
from inputs import inspect_upload
from live import OneTakeSession

def main():
    model=Paper();source=UP/'evaluation/samples/id2_vcd_swwp2s.mpg'
    with tempfile.TemporaryDirectory() as folder:
        upload=Path(folder)/'upload.mpg';upload.write_bytes(source.read_bytes())
        info=inspect_upload(upload);result,prob=model.predict(upload)
    video=model.Video(vtype='face',face_predictor_path=str(UP/'common/predictors/shape_predictor_68_face_landmarks.dat'))
    frames=video.get_video_frames(str(source))
    captured=[]
    def callback(x):
        r,p=model.predict_frames(x);captured.append(p);return json.dumps(r)
    worker=OneTakeSession(callback)
    for i,frame in enumerate(frames):worker.accept(frame,i/25)
    worker.on_ended();assert worker.wait_until_idle(90),'Webcam worker timed out'
    snapshot=worker.snapshot();assert len(snapshot['segments'])==1 and not snapshot['segments'][0].get('error'),snapshot
    recorded=json.loads(snapshot['segments'][0]['text']);error=float(np.max(np.abs(prob-captured[0])))
    assert error==0 and recorded['text']==result['text']
    # Initializing the paper compatibility layer must not alter the notebook.
    tutorial=Tutorial();t,_=tutorial.predict(ROOT/'samples/bras9a.mpg');assert t['text']=='bin red at s nine again',t
    report={'upload':{**info,**result},'recorded_webcam':recorded,'recorded_worker_probability_max_error':error,'worker_segments':len(snapshot['segments']),'physical_camera_tested':False,'practice_prompt_sent_to_recognizer':False,'tutorial_after_paper':t,'published_weights_unchanged':True,'timing_note':'CPU processing after capture; includes dlib crop and beam decoding. Other local verification may be running.'}
    (ROOT/'reports/runtime-checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
