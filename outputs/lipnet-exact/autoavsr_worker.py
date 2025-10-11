"""Local pipe worker. Webcam RGB frames stay in memory; no audio or prompts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

OUTPUT = sys.stdout
sys.stdout = sys.stderr  # Third-party diagnostics must not enter the response protocol.
STUDIO = Path(__file__).resolve().parents[1]/'lipreading-studio'
sys.path.insert(0, str(STUDIO))


def reply(value):
    OUTPUT.write(json.dumps(value)+'\n')
    OUTPUT.flush()


def main():
    try:
        import numpy as np
        from studio.pipeline import LipReader, prepare_video, track_mouth
        provenance = json.loads((STUDIO/'models/provenance.json').read_text())
        for name in ['autoavsr.pth', 'face_detector.tflite']:
            with (STUDIO/'models'/name).open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != provenance['files'][name]['sha256']:
                raise RuntimeError(f'Auto-AVSR integrity check failed: {name}')
        reader = LipReader(beam_size=5)
        reply({'ready': True, 'device': reader.device})
    except Exception as error:
        reply({'error': str(error)})
        return
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            size = request.get('bytes', 0)
            if not isinstance(size, int) or not 0 <= size <= 300_000_000:
                reply({'error': 'Invalid frame payload size'})
                return
            payload = sys.stdin.buffer.read(size)
            if len(payload) != size:
                return
            start = time.perf_counter()
            if request['kind'] == 'video':
                with tempfile.TemporaryDirectory(prefix='autoavsr-upload-') as folder:
                    path = Path(folder)/'silent.mp4'
                    prepare_video(request['path'], path)
                    result = reader.transcribe(path)
                    text = result['text']
            elif request['kind'] == 'frames':
                frames = np.load(io.BytesIO(payload), allow_pickle=False)
                if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or not 10 <= len(frames) <= 300:
                    raise ValueError('Expected 0.4–12 seconds of RGB webcam frames')
                crops, _ = track_mouth(frames)
                text = reader.recognize(crops)
            else:
                raise ValueError('Unknown inference request')
            reply({'result': {'text': text, 'raw': text, 'seconds': time.perf_counter()-start,
                             'mode': 'Auto-AVSR · unrestricted English · original pretrained weights · beam 5',
                             'checkpoint': 'autoavsr', 'device': reader.device, 'audio_used': False}})
        except Exception as error:
            reply({'error': str(error)})


if __name__ == '__main__':
    main()
