"""Read-only, loopback-only dashboard for the existing LRS3 training job."""
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {
    'lrs2': ROOT / 'outputs/lipreading-studio/training/runs/lrs2-full-v1',
    'lrs3': ROOT / 'outputs/lipreading-studio/training/runs/lrs3-only-v2',
}

def read(run, name, default=None):
    path = run / name
    if not path.exists():
        return default
    return json.loads(path.read_text())

def snapshot(key):
    run = RUNS[key]
    config = read(run, 'config.json', {})
    return dict(progress=read(run, 'progress.json', {}), history=read(run, 'history.json', []),
                baseline=read(run, 'baseline-validation.json'), selection=read(run, 'selection.json'),
                result=read(run, 'result.json'), now=time.time(),
                config={key: config.get(key) for key in ('splits', 'max_epochs', 'patience')})

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        key = parse_qs(parsed.query).get('run', ['lrs2'])[0]
        if key not in RUNS:
            self.send_error(400, 'Unknown run')
            return
        if path not in ('/', '/api/progress'):
            self.send_error(404)
            return
        try:
            body = (json.dumps(snapshot(key)).encode() if path == '/api/progress'
                    else (HERE / 'index.html').read_bytes())
        except (OSError, ValueError):
            self.send_error(503, 'Progress temporarily unavailable; retrying shortly')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json' if path == '/api/progress' else 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass

if __name__ == '__main__':
    print('Training dashboard: http://127.0.0.1:8766', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
