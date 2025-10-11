"""Keep the retained PyTorch lip reader separate from the legacy TensorFlow runtime."""
import atexit
import io
import json
import os
from pathlib import Path
import select
import subprocess
import threading

import numpy as np

ROOT = Path(__file__).resolve().parent


class AutoAVSR:
    def __init__(self):
        python = Path(os.environ.get('AUTOAVSR_PYTHON', '/private/tmp/grid-clean-env/bin/python'))
        if not python.is_file():
            raise RuntimeError('Auto-AVSR runtime is unavailable. See AUTOAVSR.md for local setup.')
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [str(python), '-u', str(ROOT/'autoavsr_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env={**os.environ, 'MPLCONFIGDIR': '/private/tmp/lipnet-exact-mpl'})
        atexit.register(self.close)
        try:
            ready = self._read()
            if not ready.get('ready'):
                raise RuntimeError(ready.get('error', 'Auto-AVSR failed to start.'))
        except Exception:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], 180)[0]:
            self.close()
            raise RuntimeError('Auto-AVSR timed out. Restart the local app and retry.')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Auto-AVSR stopped. Restart the local app and retry.')
        return json.loads(line)

    def _request(self, command, payload=b''):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('Auto-AVSR worker is stopped; restart the local app.')
            self.process.stdin.write(json.dumps({**command, 'bytes': len(payload)}).encode()+b'\n')
            if payload:
                self.process.stdin.write(payload)
            self.process.stdin.flush()
            response = self._read()
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response['result'], None

    def predict(self, path):
        return self._request({'kind': 'video', 'path': str(Path(path).resolve())})

    def predict_frames(self, frames):
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError('Expected RGB uint8 webcam frames.')
        if not 10 <= len(frames) <= 300 or max(frames.shape[1:3]) > 640:
            raise ValueError('Webcam segments must be 0.4–12 seconds and at most 640 pixels per side.')
        payload = io.BytesIO()
        np.save(payload, frames, allow_pickle=False)
        return self._request({'kind': 'frames'}, payload.getvalue())

    def close(self):
        process = getattr(self, 'process', None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
