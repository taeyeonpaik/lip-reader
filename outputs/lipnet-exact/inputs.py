"""Bound upload decoding without changing source pixels or frame rate."""
import json,subprocess
from fractions import Fraction

def inspect_upload(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],timeout=15))
    video=next((v for v in info.get('streams',[]) if v.get('codec_type')=='video'),None)
    if video is None:raise ValueError('The file has no video stream.')
    duration=float(info.get('format',{}).get('duration',video.get('duration',0)))
    if not 1<=duration<=5:raise ValueError('Use one sentence lasting 1–5 seconds.')
    if max(int(video['width']),int(video['height']))>640:raise ValueError('Export the clip with its longest side at most 640 pixels (360 × 288 is the original GRID size).')
    rate=float(Fraction(video.get('avg_frame_rate','0')))
    if abs(rate-25)>.1:raise ValueError('Export at 25 frames per second to match the original model.')
    return {'duration_seconds':duration,'fps':rate,'width':video['width'],'height':video['height']}
