"""Cross-check processed LRS3 identities/transcripts against an independent label list."""
import argparse,hashlib,json
from pathlib import Path
from training.metrics import normalize
from training.data import atomic_json

def verify(manifest,labels,destination):
    records={}
    for line in labels.read_text().splitlines():
        _,path,frames,_,text=line.split(',',4)
        identity=path.split('/trainval/')[1].removesuffix('.mp4')
        records[identity]={'frames':int(frames),'text':normalize(text)}
    rows=[r for r in json.loads(manifest.read_text()) if r['dataset']=='LRS3-TED']
    problems=[]
    for row in rows:
        identity=row['id'].removeprefix('lrs3/');ref=records.get(identity)
        if ref is None or ref['frames']!=row['declared_frames'] or ref['text']!=row['text']:
            problems.append({'id':identity,'expected':ref,'imported_frames':row['declared_frames'],'imported_text':row['text']})
    report={'state':'passed' if not problems else 'failed','checked_clips':len(rows),'mismatches':problems,'reference':'Llama-AVSR authors public 30-hour LRS3 label list','reference_url':'https://drive.google.com/file/d/11t5BDnl05p3A5kR1rIaSosW2vXK915BN/view','reference_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),'note':'Processed distributor metadata has generic source/corpus_tier fields inconsistent with its README. Clip IDs, frame counts, and normalized transcripts are verified against this independent author label list; those generic fields are not used for splits or corpus identity.'}
    atomic_json(destination,report)
    if problems:raise ValueError(f'{len(problems)} LRS3 annotation mismatches; inspect before use')
    print(json.dumps({k:v for k,v in report.items() if k!='mismatches'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','labels','destination']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();verify(a.manifest,a.labels,a.destination)
