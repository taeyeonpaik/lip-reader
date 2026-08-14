"""Independent CPU data audit; no model inference or test-based selection."""
import collections, hashlib, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2
ROOT=pathlib.Path('/teamspace/studios/this_studio/lipreader')
P=ROOT/'work/corpora/tcd-timit'
sys.path.insert(0,str(ROOT/'outputs/lipreading-studio'))
from training.tcd_protocol import validate

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def check(r):
    assert sha(r['path'])==r['sha256'],r['id']
    cap=cv2.VideoCapture(r['path']);assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01
    n=0;digest=hashlib.sha256()
    while True:
        ok,f=cap.read()
        if not ok:break
        assert f.shape==(96,96,3)
        digest.update(cv2.cvtColor(f,cv2.COLOR_BGR2RGB).tobytes());n+=1
    cap.release();assert n==r['declared_frames']>0
    return digest.hexdigest()

def main():
    cv2.setNumThreads(1)
    rows=json.loads((P/'manifest.json').read_text());source=json.loads((P/'source-inventory.json').read_text())
    failures=json.loads((P/'preparation-failures.json').read_text())
    original={r['id']:r for r in source['rows']};prepared={r['id']:r for r in rows}
    assert len(original)==5782 and len(prepared)==len(rows)==5781
    assert {r['id'] for r in failures}=={'volunteer/56M/sx67'}
    assert set(prepared)|{r['id'] for r in failures}==set(original)
    assert not set(prepared)&{r['id'] for r in failures}
    for r in rows:
        s=original[r['id']]
        assert all(r[k]==s[k] for k in ('text','split','speaker','transcript_sha256','population','view'))
        words=[]
        for line in s['original_transcript'].splitlines():
            start,end,word=line.split(maxsplit=2);assert int(end)>=int(start);words.append(word)
        import re
        ref=' '.join(re.sub(r"[^A-Z0-9' ]",' ',' '.join(words).upper()).split())
        assert ref==r['text'],r['id']
    evaluation_failures=[]
    for f in failures:
        s=original[f['id']];assert f['split']==s['split']=='test'
        evaluation_failures.append(dict(id=s['id'],text=s['text'],split=s['split'],speaker=s['speaker'],population=s['population'],session='volunteer:'+s['speaker'],evaluation_group='volunteer:'+s['speaker'],preprocessing_error=f['error']))
    provenance=json.loads((P/'provenance.json').read_text())
    candidate=dict(provenance,review_complete=True)
    report=validate(rows,candidate)
    seen={};start=time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(r,d) in enumerate(zip(rows,pool.map(check,rows)),1):
            assert d not in seen or seen[d]==r['split'],'Identical decoded video across splits'
            seen[d]=r['split']
            if i%100==0 or i==len(rows):
                save(P/'audit-progress.json',dict(complete=i,total=len(rows),elapsed_seconds=time.time()-start))
                print(f'Audited {i}/{len(rows)}',flush=True)
    report.update(passed=True,sha256_verified=True,decoded_verified=True,reference_audit=True,
                  total_source_clips=len(original),evaluation_failures=evaluation_failures,
                  source_manifest_sha256=sha(P/'manifest.json'),source_inventory_sha256=sha(P/'source-inventory.json'),
                  final_test_denominator=882,test_failure_policy='Empty transcript; all reference words count as deletions for baseline and selected model',
                  shared_texts_across_splits=source['report']['shared_texts_across_splits'],official_split=False)
    save(P/'evaluation-failures.json',evaluation_failures)
    save(P/'independent-data-audit.json',report)
    provenance.update(review_complete=True,complete=False,ready_for_training=True,failed_test_clips_accounted=True,review='independent-data-audit.json')
    save(P/'provenance.json',provenance)
    print('DATA AUDIT PASSED; 5781 prepared clips plus 1 recorded test failure; full test denominator 882.',flush=True)
if __name__=='__main__':main()
