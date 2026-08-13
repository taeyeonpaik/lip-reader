"""Resumable CPU preparation of licensed frontal volunteer recordings."""
import argparse, collections, fcntl, hashlib, json, os, random, re, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
ROOT = Path('/teamspace/studios/this_studio/lipreader')
PROJECT = ROOT/'outputs/lipreading-studio'
sys.path.insert(0,str(PROJECT))

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def parse_transcript(original):
    words=[]
    for line in original.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(.+?)\s*",line)
        if not match or int(match[2])<int(match[1]):
            raise ValueError('Expected timestamped word-level transcript')
        words.append(match[3])
    text=' '.join(re.sub(r"[^A-Z0-9' ]",' ',' '.join(words).upper()).split())
    if not text:raise ValueError('Empty word reference')
    return text

def inventory(dest, limit=None):
    raw=ROOT/'data/tcd-timit'
    archives=sorted((raw/'volunteers').glob('*/Clips/straightcam.zip'))
    assert len(archives)==59, f'Expected 59 volunteers, got {len(archives)}'
    speakers=[p.parents[1].name for p in archives]
    rng=random.Random(91); shuffled=sorted(speakers);rng.shuffle(shuffled)
    splits={s:part for part,group in [('train',shuffled[:41]),('validation',shuffled[41:50]),('test',shuffled[50:])] for s in group}
    rows=[];sources=[]
    for path in archives:
        speaker=path.parents[1].name
        receipt=json.loads(path.with_name(path.name+'.download.json').read_text())
        assert path.stat().st_size==receipt['bytes'],str(path)
        sources.append(dict(path=str(path),bytes=receipt['bytes'],sha256=receipt['sha256']))
        cache=dest/'inventory-cache'/(speaker+'.json')
        if cache.exists():
            prior=json.loads(cache.read_text())
            assert prior['archive_sha256']==receipt['sha256']
            for r in prior['rows']:r['text']=parse_transcript(r['original_transcript'])
            rows.extend(prior['rows'])
            print('Inventory cached: '+speaker,flush=True)
            if limit and len(rows)>=limit:break
            continue
        first=len(rows)
        with zipfile.ZipFile(path) as z:
            names={n.lower():n for n in z.namelist()}
            for name in sorted(z.namelist()):
                if not name.lower().endswith('.mp4'):continue
                stem=Path(name).stem.lower()
                textname=names.get(str(Path(name).with_suffix('.txt')).lower())
                assert textname, f'Missing text: {speaker}/{stem}'
                original=z.read(textname).decode('utf-8-sig').strip()
                # TIMIT word references may start with start/end sample offsets.
                text=parse_transcript(original)
                assert text and any(c.isalpha() for c in text),original
                rows.append(dict(id=f'volunteer/{speaker}/{stem}',speaker=speaker,population='volunteer',view='straightcam',split=splits[speaker],archive=str(path),member=name,text=text,original_transcript=original,transcript_source=f'{receipt["url"]}#{textname}',transcript_sha256=hashlib.sha256(z.read(textname)).hexdigest()))
        save(cache,dict(archive_sha256=receipt['sha256'],rows=rows[first:]))
        print(f'Inventoried {speaker}: {len(rows)-first} clips',flush=True)
        save(dest/'inventory-progress.json',dict(speakers_complete=len(sources),total_speakers=59,clips=len(rows)))
        if limit and len(rows)>=limit:break
    texts=collections.defaultdict(set)
    for r in rows:texts[r['text']].add(r['split'])
    report=dict(dataset='TCD-TIMIT',speakers={part:sorted(s for s in speakers if splits[s]==part) for part in ['train','validation','test']},clips=dict(collections.Counter(r['split'] for r in rows)),total=len(rows),split_policy='Custom 41/9/9 speaker split; sorted speaker IDs shuffled with Python random.Random(91); not an official benchmark split',shared_texts_across_splits=sum(len(v)>1 for v in texts.values()),limitation='Speaker-disjoint, not guaranteed sentence-disjoint; shared read prompts are reported, not used for selection.',excluded_archives=['Lipspkr1','Lipspkr2','Lipspkr3'],exclusion_reason='HTTP 503 after bounded retries; user approved volunteer-only run',source_archives=sources)
    pinned=dest/('pilot-source-inventory.json' if limit else 'source-inventory.json')
    value=dict(report=report,rows=rows)
    if pinned.exists():assert json.loads(pinned.read_text())==value,'Pinned inventory changed'
    else:save(pinned,value)
    return rows,report

def convert(row,dest):
    import cv2,numpy as np
    from studio.pipeline import prepare_video,track_mouth,frame_chunks
    out=dest/'videos'/row['speaker']/(Path(row['member']).stem+'.mp4')
    record=out.with_suffix('.json')
    if record.exists():
        cached=json.loads(record.read_text())
        assert cached['preparation_sha256']==sha(__file__),'Preparation source changed'
        assert cached['sha256']==sha(out),'Prepared video hash changed'
        return cached
    if shutil.disk_usage(dest).free<12*1024**3:raise RuntimeError('Less than 12 GiB free; stop preparation')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tcd-',dir=dest) as tmp:
        raw=Path(tmp)/'source.mp4';normal=Path(tmp)/'normalized.mp4'
        with zipfile.ZipFile(row['archive']) as z, z.open(row['member']) as source, raw.open('wb') as target:
            shutil.copyfileobj(source,target,4*1024*1024)
        source_sha=sha(raw)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(raw)]))
        v=next(s for s in probe['streams'] if s['codec_type']=='video')
        duration=prepare_video(raw,normal,max_seconds=120)
        frames=np.concatenate([f for _,f in frame_chunks(normal,seconds=120)])
        crops,coverage=track_mouth(frames)
        tmpout=out.with_name(out.stem+'.tmp.mp4')
        cmd=['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s','96x96','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p',str(tmpout)]
        subprocess.run(cmd,input=crops.tobytes(),check=True,timeout=120)
        cap=cv2.VideoCapture(str(tmpout));count=0
        assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01
        while True:
            ok,f=cap.read()
            if not ok:break
            assert f.shape==(96,96,3);count+=1
        cap.release();assert count==len(crops)>0
        tmpout.replace(out)
        result={k:v for k,v in row.items() if k not in ('archive','member','original_transcript')}
        result.update(path=str(out),declared_frames=count,source_sha256=source_sha,sha256=sha(out),duration_seconds=count/25,source_duration_seconds=duration,source_fps=v['r_frame_rate'],tracking_coverage=coverage,preparation_sha256=sha(__file__))
        save(record,result)
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory-only',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
    dest=ROOT/'work/corpora/tcd-timit';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows,report=inventory(dest,a.limit);print(json.dumps({k:v for k,v in report.items() if k!='source_archives'}),flush=True)
        if a.inventory_only:return
        for binary in ('ffmpeg','ffprobe'):
            subprocess.run([binary,'-version'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        import torch,cv2,mediapipe
        torch.set_num_threads(2);cv2.setNumThreads(1)
        selected=rows[:a.limit] if a.limit else rows
        completed=[];failures=[];start=time.time()
        for row in selected:
            try:completed.append(convert(row,dest))
            except Exception as e:
                failures.append(dict(id=row['id'],split=row['split'],error=str(e)))
                print('FAILED '+row['id']+': '+str(e),flush=True)
                if isinstance(e,RuntimeError) and 'GiB' in str(e):raise
            save(dest/'preparation-progress.json',dict(total=len(rows),scheduled=len(selected),complete=len(completed),failed=len(failures),last_id=row['id'],elapsed_seconds=time.time()-start))
            save(dest/'preparation-failures.json',failures)
            print(f'Prepared {len(completed)}/{len(selected)}; failures {len(failures)}; {row["id"]}',flush=True)
        save(dest/('pilot-manifest.json' if a.limit else 'manifest.json'),completed)
        save(dest/'provenance.json',dict(dataset='TCD-TIMIT',source='Official authenticated TCD-TIMIT frontal volunteer archives',transcript_source='Word transcript TXT bundled alongside each video; no generated references',preprocessing='App pipeline: ffmpeg 25fps <=640px RGB, MediaPipe face tracking, author VideoProcess 96x96 RGB, H264 CRF18; not exact original benchmark preprocessing',split_policy=report['split_policy'],review_complete=False,exclusions=report['excluded_archives'],preparation_sha256=sha(__file__),complete=not a.limit and not failures,failures=len(failures)))
        print('Preparation finished; independent review required before training.',flush=True)
if __name__=='__main__':main()
