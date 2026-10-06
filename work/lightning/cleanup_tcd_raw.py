"""Preserve TCD references and failed source before removing raw ZIP/partial files.
Run on the original Studio. Default is a fully verified plan; --apply deletes only
listed raw archives/partials after all preservation checks succeed.
"""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
ROOT=Path('/teamspace/studios/this_studio/lipreader')
RAW=ROOT/'data/tcd-timit'
PREP=ROOT/'work/corpora/tcd-timit'
KEEP=ROOT/'work/tcd-raw-preserved'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def safe_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert sha(source)==sha(target),str(source)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    audit=json.loads((PREP/'independent-data-audit.json').read_text())
    assert audit['passed'] and sha(PREP/'manifest.json')==audit['source_manifest_sha256']
    assert sha(PREP/'source-inventory.json')==audit['source_inventory_sha256']
    rows=json.loads((PREP/'manifest.json').read_text())
    assert len(rows)==5781
    for r in rows:
        path=Path(r['path']);assert path.resolve().is_relative_to(PREP.resolve())
        assert sha(path)==r['sha256'],'Prepared crop changed: '+r['id']
    run=ROOT/'outputs/lipreading-studio/training/runs/tcd-timit-v1'
    selection=json.loads((run/'selection.json').read_text())
    best=Path(selection['checkpoint'])
    assert best.is_file() and sha(best)==selection['sha256'],'Selected trained checkpoint not verified'
    sources=json.loads((PREP/'source-inventory.json').read_text())
    archives=sorted(RAW.glob('volunteers/*/Clips/straightcam.zip'))
    assert len(archives)==59,'Expected all original volunteer archives before cleanup'
    expected={r['id']:r for r in sources['rows']}
    preserved=[]
    for archive in archives:
        assert not archive.is_symlink()
        speaker=archive.parents[1].name
        with zipfile.ZipFile(archive) as z:
            names={n.lower():n for n in z.namelist()}
            for ident,row in expected.items():
                if row['speaker']!=speaker:continue
                member=names[str(Path(row['member']).with_suffix('.txt')).lower()]
                data=z.read(member)
                assert hashlib.sha256(data).hexdigest()==row['transcript_sha256'],ident
                dest=KEEP/'transcripts'/speaker/Path(member).name
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                assert sha(dest)==row['transcript_sha256']
                preserved.append(ident)
            if speaker=='56M':
                member=expected['volunteer/56M/sx67']['member']
                dest=KEEP/'failed-test-source/56M/sx67.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256()
                with z.open(member) as src,dest.open('wb') as out:
                    while chunk:=src.read(4*1024*1024):out.write(chunk);h.update(chunk)
                assert sha(dest)==h.hexdigest() and dest.stat().st_size>0
                save(dest.with_suffix('.provenance.json'),dict(source_archive=str(archive),member=member,sha256=h.hexdigest()))
    assert set(preserved)==set(expected) and len(preserved)==5782
    # Retain small original labels, receipts, and download logs separately.
    for p in RAW.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.txt','.mlf','.json'):
            safe_copy(p,KEEP/'download-metadata'/p.relative_to(RAW))
    for name in ('manifest.json','source-inventory.json','provenance.json','independent-data-audit.json','evaluation-failures.json'):
        safe_copy(PREP/name,KEEP/'prepared-metadata'/name)
    safe_copy(run/'selection.json',KEEP/'trained-selection.json')
    targets=archives+sorted(RAW.glob('lipspeakers/*/Clips/straightcam.zip.part'))
    assert all(p.resolve().is_relative_to(RAW.resolve()) and not p.is_symlink() for p in targets)
    plan=dict(transcripts_preserved=len(preserved),trained_checkpoint_verified=str(best),
              failed_source_preserved=str(KEEP/'failed-test-source/56M/sx67.mp4'),
              deletion_targets=[dict(path=str(p),bytes=p.stat().st_size) for p in targets],
              total_bytes=sum(p.stat().st_size for p in targets),applied=False)
    save(KEEP/'cleanup-plan.json',plan)
    print(f'Preservation verified; {len(targets)} raw files, {plan["total_bytes"]/2**30:.2f} GiB removable.',flush=True)
    if a.apply:
        deleted=[]
        for item in plan['deletion_targets']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes']
            p.unlink();deleted.append(item)
            save(KEEP/'cleanup-deleted.json',deleted)
        plan['applied']=True;save(KEEP/'cleanup-plan.json',plan)
        print('Raw cleanup complete. Prepared crops, checkpoints and preserved references retained.',flush=True)
if __name__=='__main__':main()
