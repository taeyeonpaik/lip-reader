"""Tutorial-compatible inference and explicitly separate automatic webcam alignment."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,threading,time
import cv2,numpy as np,torch
from .accelerated import AcceleratedLipNet,greedy_decode
from .grammar import constrained_decode
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_rgb(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,f=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if not frames:raise ValueError('No video frames were read.')
    return np.stack(frames)

def normalize_crops(crops):
    # Preserve the tutorial's BGR-as-RGB conversion and uint8 centering behavior.
    bgr=crops[...,::-1].astype(np.float32)/255.
    gray=np.minimum((bgr@np.array([.2989,.587,.114],np.float32))*255.5,255).astype(np.uint8)
    center=int(np.mean(gray,dtype=np.float64))
    std=float(np.std(gray.astype(np.float32)))
    if std<1e-6:raise ValueError('Mouth frames have no visual variation.')
    return ((gray-np.uint8(center)).astype(np.float32)/std)[...,None]

def fixed_input(frames):
    if frames.shape[1]<236 or frames.shape[2]<220:raise ValueError('Tutorial mode requires original GRID frame dimensions.')
    if not 74<=len(frames)<=75:raise ValueError('Tutorial examples must contain 74 or 75 frames.')
    x=normalize_crops(frames[:,190:236,80:220])
    return np.pad(x,((0,75-len(x)),(0,0),(0,0),(0,0)))

def detections(frames):
    import mediapipe as mp
    options=mp.tasks.vision.FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/face_detector.tflite')),running_mode=mp.tasks.vision.RunningMode.VIDEO,min_detection_confidence=.7)
    points=[]
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for i,frame in enumerate(frames):
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(frame)),i*40)
            if len(result.detections)>1:raise ValueError('Show only one face.')
            if not result.detections:points.append(None);continue
            d=result.detections[0]
            if min(d.bounding_box.width,d.bounding_box.height)<60:raise ValueError('Move closer to the camera.')
            h,w=frame.shape[:2];points.append(np.array([[d.keypoints[k].x*w,d.keypoints[k].y*h] for k in [0,1,3]],np.float32))
    return points

def aligned_input(frames):
    if not 25<=len(frames)<=125:raise ValueError('Record one sentence lasting 1–5 seconds.')
    target=np.array(json.loads((ROOT/'models/face-template.json').read_text())['points'],np.float32)
    points=detections(frames);known=[i for i,p in enumerate(points) if p is not None]
    if len(known)<.8*len(frames):raise ValueError('Keep one well-lit face visible throughout the sentence.')
    if max(np.diff([-1]+known+[len(frames)]))>13:raise ValueError('Face tracking was interrupted. Try again.')
    values=np.stack([points[i] for i in known]);smooth=np.empty((len(frames),3,2),np.float32)
    for k in range(3):
        for d in range(2):smooth[:,k,d]=np.interp(np.arange(len(frames)),known,values[:,k,d])
    crops=[]
    for i,frame in enumerate(frames):
        src=smooth[max(0,i-2):min(len(frames),i+3)].mean(0)
        matrix=cv2.getAffineTransform(src,target)
        aligned=cv2.warpAffine(frame,matrix,(360,288),borderMode=cv2.BORDER_REPLICATE)
        crops.append(aligned[190:236,80:220])
    # Tutorial consumes a fixed three-second sequence; report this resampling.
    crops=np.stack(crops)[np.rint(np.linspace(0,len(crops)-1,75)).astype(int)]
    return normalize_crops(crops)

def prepare_upload(source,destination):
    probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(source)],capture_output=True,timeout=20)
    if probe.returncode:raise ValueError('The upload is not a readable video.')
    info=json.loads(probe.stdout);streams=[s for s in info['streams'] if s['codec_type']=='video']
    if not streams:raise ValueError('No video track found.')
    duration=float(info['format'].get('duration',0))
    if not 1<=duration<=5:raise ValueError('Upload one sentence lasting 1–5 seconds.')
    if max(streams[0]['width'],streams[0]['height'])>4096:raise ValueError('Export the video at 4K or below.')
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(source),'-map','0:v:0','-an','-sn','-dn','-vf','fps=25,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2','-c:v','libx264','-pix_fmt','yuv420p',str(destination)],check=True,capture_output=True,timeout=60)

class GridReader:
    def __init__(self,checkpoint=None):
        torch.set_num_threads(4);self.device='mps' if torch.backends.mps.is_available() else 'cpu';self.lock=threading.Lock()
        self.model=AcceleratedLipNet();self.model.load_keras_arrays(ROOT/'models/tutorial-published.npz')
        self.origin='Tutorial author checkpoint (96 epochs)'
        if checkpoint:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved.get('format')!='grid-tutorial-v1':raise ValueError('Wrong checkpoint format')
            self.model.load_state_dict(saved['state_dict'],strict=True);self.origin='Locally adapted tutorial model'
        self.model.to(self.device).eval()
    def logits(self,x):
        with self.lock,torch.inference_mode():return self.model(torch.from_numpy(x[None]).to(self.device))[0].cpu()
    def recognize(self,frames,fixed=False):
        start=time.perf_counter();x=fixed_input(frames) if fixed else aligned_input(frames);logits=self.logits(x)
        return {'raw':greedy_decode(logits[None])[0],'constrained':constrained_decode(logits),'seconds':time.perf_counter()-start,'model':self.origin,'mode':'original GRID crop' if fixed else 'automatic face alignment; temporal resampling','audio_used':False}
