"""Compare upload and paced webcam-worker throughput using included video only."""
from pathlib import Path
import json, sys, tempfile, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from studio.pipeline import ROOT, LipReader, prepare_video, track_mouth
from studio.live import LiveSession

reader = LipReader()
source = ROOT/'examples/research-demo.mp4'
frame_temp = tempfile.TemporaryDirectory()
frame_source = Path(frame_temp.name)/'frames.mp4'
prepare_video(source, frame_source)
cap = cv2.VideoCapture(str(frame_source)); frames = []
while True:
    ok, frame = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
frame_temp.cleanup()
assert len(frames) == 200
report = {'physical_camera_tested':False,'audio_used':False,'note':'One eight-second included example; two upload repeats and two paced webcam windows per decoder. No general throughput guarantee. Model loading excluded; uploads include normalization, tracking and recognition.','profiles':{}}
for beam in [5,20]:
    reader.configure_decoding(beam_size=beam)
    uploads = []
    for repeat in range(2):
        with tempfile.TemporaryDirectory() as temp:
            tick = time.monotonic(); normalized = Path(temp)/'normalized.mp4'
            prepare_video(source, normalized); result = reader.transcribe(normalized)
            uploads.append({'total_seconds':time.monotonic()-tick, **result})
    live = LiveSession(lambda batch:reader.recognize(track_mouth(batch)[0]))
    tick = time.monotonic()
    for i in range(400):
        time.sleep(max(0,tick+i/25-time.monotonic()))
        live.accept(frames[i%200],i/25)
    live.on_ended()
    assert live.wait_until_idle(60)
    state = live.snapshot()
    assert len(state['segments']) == 2 and state['skipped'] == 0, state
    assert all(s['text'] and not s.get('error') for s in state['segments']), state
    report['profiles'][str(beam)] = {'uploads':uploads,'paced_webcam':state,'capture_and_drain_seconds':time.monotonic()-tick}
    print('beam',beam,'complete',flush=True)
(ROOT/'validation/decoding-throughput.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
