from pathlib import Path
import hashlib,json,time
import gdown
root=Path(__file__).resolve().parents[2]
p=root/'outputs/lipreading-studio/models/autoavsr.pth'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
expected='fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb'
if not p.exists() or sha(p)!=expected:
 tmp=p.with_suffix('.download')
 result=gdown.download(id='1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_',output=str(tmp),quiet=False)
 assert result and sha(tmp)==expected,'Original checkpoint download/hash failed'
 tmp.replace(p)
(root/'work/lightning/weight-transfer-verification.json').write_text(json.dumps(dict(sha256=sha(p),verified=True,completed_unix=time.time()),indent=2)+'\n')
print('Original author checkpoint SHA256 verified',flush=True)
