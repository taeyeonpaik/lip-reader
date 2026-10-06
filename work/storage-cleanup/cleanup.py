from pathlib import Path
import os,json,shutil,time
root=Path.cwd().resolve()
targets=['work/deeper/features','work/deeper/dataset','work/overnight/data','work/overnight/dataset','work/overnight/features','work/overnight/smoke','work/tf-venv','work/tf-metal-venv','work/venv','work/training-cache','work/paper-training-cache','work/tutorial-assets/dataset','work/grid-return/cache','work/videomae/features','work/videomae/pilot','work/paper-full-smoke','work/tutorial-training-check/checkpoints','work/lipnet-exact/dlib-20.0.1','work/lipnet-exact-rizkiarm','outputs/lipreading-studio.zip','outputs/lipnet-from-scratch.zip','outputs/lipnet-grid-practice.zip','outputs/lipnet-video-tutorial.zip','outputs/lipnet-video-tutorial 2.zip']
# Completed, rejected runs: retain code, metrics and predictions, remove optimizer resumes.
for d in ['outputs/lipreading-studio/training/runs','outputs/lipnet-tutorial/runs','outputs/lipnet/runs']:
    targets.extend(str(p) for p in Path(d).rglob('*') if p.is_file() and p.suffix in ['.pt','.pth','.h5'])
report={'authorized_by':'User requested deletion of large unused project files','started':time.time(),'preserved':['outputs/lipnet-exact','work/grid-return/data','work/corpora','original published model files','historical source, metrics and prediction reports'],'removed':[],'errors':[]}
report['free_disk_before']=shutil.disk_usage(root).free
out=root/'outputs/STORAGE_CLEANUP.json'
def save():out.write_text(json.dumps(report,indent=2)+'\n')
for rel in targets:
    p=root/rel
    assert p.is_relative_to(root) and rel.startswith(('work/','outputs/'))
    if not p.exists():continue
    logical=allocated=count=0
    paths=[p] if p.is_file() else (Path(d)/n for d,_,names in os.walk(p) for n in names)
    for f in paths:
        try:s=f.lstat();logical+=s.st_size;allocated+=s.st_blocks*512;count+=1
        except OSError:pass
    try:
        if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
        else:p.unlink()
        report['removed'].append({'path':rel,'logical_bytes':logical,'allocated_bytes_before':allocated,'files':count});print(rel,round(logical/2**30,3),'GiB',flush=True)
    except OSError as e:report['errors'].append({'path':rel,'error':str(e)})
    save()
report['free_disk_after']=shutil.disk_usage(root).free;report['total_logical_bytes_removed']=sum(x['logical_bytes'] for x in report['removed']);report['completed']=time.time();save();print('COMPLETE',report['total_logical_bytes_removed']/2**30,flush=True)
