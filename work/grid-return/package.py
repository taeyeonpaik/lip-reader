"""Build current GRID prototype without full datasets or optimizer checkpoints."""
from pathlib import Path
import hashlib,json,zipfile
root=Path('outputs/lipnet-tutorial');dest=Path('outputs/lipnet-grid-practice.zip')
files=[root/p for p in ['app.py','README.md','REVIEW.md','requirements-grid.txt','Start LipNet.command','training/GRID_RETURN.md','training/grid_experiment.py','training/review_grid.py','tutorial/__init__.py','tutorial/accelerated.py','tutorial/runtime.py','tutorial/grammar.py','tutorial/live.py','models/tutorial-published.npz','models/face_detector.tflite','models/face-template.json','models/grid-provenance.json']]
files+=list((root/'tests_grid').glob('*.py'))+list((root/'runs/grid-return-v1').glob('*.json'))+list((root/'validation/grid').glob('*'))
files+=[p for p in (root/'samples').glob('*/*') if p.suffix in ['.mpg','.mp4','.align']]
files+=[p for p in [root/'models/grid-adapted.pt',root/'models/grid-selection.json'] if p.exists()]
manifest={}
with zipfile.ZipFile(dest.with_suffix('.tmp'), 'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in sorted(set(files)):
  if not p.is_file():raise FileNotFoundError(p)
  data=p.read_bytes()
  if len(data)!=p.stat().st_size:raise RuntimeError('Incomplete/cloud-placeholder read: '+str(p))
  name='lipnet-grid-practice/'+str(p.relative_to(root));z.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
 z.writestr('lipnet-grid-practice/FILE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dest.with_suffix('.tmp')) as z:
 assert z.testzip() is None
 for name,sha in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
 assert not any('/data/' in n or n.endswith('last.pt') for n in z.namelist())
dest.with_suffix('.tmp').replace(dest)
report={'archive':str(dest),'files':len(manifest)+1,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'training_complete':(root/'runs/grid-return-v1/result.json').exists(),'contains_full_dataset':False,'zip_crc_and_member_sha256_verified':True}
Path('outputs/GRID_PACKAGE_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
