from pathlib import Path
import json,hashlib,zipfile,datetime
root=Path('outputs/lipreading-studio');output=Path('outputs/lipreading-studio.zip')
# Keep final trained heads and optimizer for reproducibility. Exclude caches/locks.
files=[p for p in root.rglob('*') if p.is_file() and not any(part in {'__pycache__','.pytest_cache','.venv'} for part in p.parts) and p.name not in {'.DS_Store','.lock'} and p.suffix not in {'.pyc','.tmp'}]
archive=output.with_suffix('.tmp')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in files:z.write(p,p.relative_to(root.parent))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert 'lipreading-studio/training/runs/lrs2-heads-v1/result.json' in z.namelist()
 assert 'lipreading-studio/models/autoavsr.pth' in z.namelist()
 assert 'lipreading-studio/training/runs/lrs2-deeper-v2/result.json' in z.namelist()
 assert 'lipreading-studio/training/runs/lrs2-deeper-v2/best.pt' in z.namelist()
 assert 'lipreading-studio/validation/deeper-checkpoint-and-metrics.json' in z.namelist()
 assert 'lipreading-studio/training/runs/lrs2-lrs3-mixed-v4/last.pt' in z.namelist()
 assert 'lipreading-studio/training/runs/lrs2-lrs3-mixed-v4/result.json' in z.namelist()
 assert 'lipreading-studio/training/runs/decoding-v3/cross-dataset-result.json' in z.namelist()
 assert 'lipreading-studio/validation/mixed-checkpoint-and-metrics.json' in z.namelist()
 assert 'lipreading-studio/validation/decoding-throughput.json' in z.namelist()
 assert 'lipreading-studio/training/combined-data-provenance.json' in z.namelist()
 assert 'lipreading-studio/training/runs/mwer-v5/result.json' in z.namelist()
 assert 'lipreading-studio/training/runs/mwer-v5/mwer-epoch-02.pt' in z.namelist()
 assert 'lipreading-studio/training/runs/mwer-v5/ce-epoch-02.pt' in z.namelist()
 assert 'lipreading-studio/validation/mwer-checkpoint-and-metrics.json' in z.namelist()
 if (root/'training/runs/videomae-v6/config.json').exists():
  for name in ['training/runs/videomae-v6/result.json','training/runs/videomae-v6/verification.json','models/videomae-ctc.pt','models/videomae-small-kinetics.pt','models/videomae-provenance.json','validation/videomae-completion.json','training/VIDEOMAE_EXPERIMENT.md']:
   assert 'lipreading-studio/'+name in z.namelist(),name
archive.replace(output)
with output.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
report={'archive':output.name,'files':len(files),'bytes':output.stat().st_size,'sha256':digest,'zip_crc_check':'passed','includes_trained_checkpoints':True,'includes_dataset_videos':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
Path('outputs/PACKAGE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
