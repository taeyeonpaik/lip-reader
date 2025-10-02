from pathlib import Path
import json,time,hashlib,zipfile,subprocess,os
root=Path.cwd();r=Path('/private/tmp/lipnet-paper-finetune-v1');archive=r/'s4.zip';out=root/'outputs/lipnet-exact/runs/paper-finetune-v1/progress.json'
def status(stage,**more):
    v={'stage':stage,'updated_unix':time.time(),**more};tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(out);print(json.dumps(v),flush=True)
status('recovering_local_training_data',completed=archive.stat().st_size if archive.exists() else 0,total=491885378)
for _ in range(180):
    size=archive.stat().st_size if archive.exists() else 0
    if size==491885378:break
    if size>491885378:raise RuntimeError('Invalid archive size')
    time.sleep(10)
else:raise RuntimeError('Archive recovery timed out')
with archive.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='1bb4543ca0a27fe76e2874845468a016'
data=r/'data';count=0
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        p=Path(item.filename)
        if p.parts[0]=='s4' and p.suffix=='.mpg':
            target=data/p;assert target.resolve().is_relative_to(data);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item));count+=1
assert count==1000
record={'s4_archive_md5':'1bb4543ca0a27fe76e2874845468a016','source':'https://zenodo.org/api/records/3625687/files/s4.zip/content','data_root':str(data),'speaker_counts':{s:len(list((data/s).glob('*.mpg'))) for s in ['s1','s2','s3','s4']},'original_ambiguous_s5_excluded':True}
(root/'outputs/lipnet-exact/runs/paper-finetune-v1/data-recovery.json').write_text(json.dumps(record,indent=2)+'\n')
assert all(v==1000 for v in record['speaker_counts'].values())
status('starting_finetuning_process')
os.execv('/private/tmp/lipnet-exact-tf/bin/python',['/private/tmp/lipnet-exact-tf/bin/python','-u','outputs/lipnet-exact/finetune.py'])
