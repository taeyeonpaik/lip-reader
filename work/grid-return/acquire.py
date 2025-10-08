from pathlib import Path
import json,subprocess,hashlib,zipfile,shutil,concurrent.futures
root=Path('work/grid-return'); dest=root/'data';dest.mkdir(exist_ok=True)
metadata=json.loads((root/'zenodo-record.json').read_text())
files=[f for f in metadata['files'] if f['key'] in ['s2.zip','s3.zip','s4.zip','s5.zip','alignments.zip']]
def acquire(f):
 p=root/f['key'];tmp=p.with_suffix('.part')
 if not p.exists():
  if shutil.disk_usage(root).free<6*1024**3:raise RuntimeError('Insufficient disk space')
  subprocess.run(['curl','-L','--fail','--retry','3','--connect-timeout','30','--max-time','1800',f['links']['self'],'-o',str(tmp)],check=True)
  tmp.replace(p)
 with p.open('rb') as stream:digest=hashlib.file_digest(stream,'md5').hexdigest()
 assert 'md5:'+digest==f['checksum'],f['key']
 with zipfile.ZipFile(p) as z:
  for name in z.namelist():
   target=(dest/name).resolve()
   if not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
  z.extractall(dest)
 print(json.dumps({'file':f['key'],'verified':True,'bytes':p.stat().st_size}),flush=True)
 return {k:f[k] for k in ['key','size','checksum','links']}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(acquire,files))
(root/'acquisition.json').write_text(json.dumps({'state':'complete','source':'https://zenodo.org/records/3625687','files':out},indent=2)+'\n')
