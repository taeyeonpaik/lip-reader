"""Acquire pinned public LRS2 main and LRS3 trainval mouth-crop releases."""
import argparse, concurrent.futures, hashlib, io, json, shutil, tarfile
from pathlib import Path
import numpy as np
import requests
from training import download as lrs2
from training.data import atomic_json, import_data, validate_splits
from training.metrics import normalize

LRS3_REPO='TheNHz/ellipsis-lrs3-trainval-autoavsr'
LRS3_REV='a9b5374e19082b3217171be8ece589f35c8a2a71'

def acquire(root,lrs2_downloads,existing_videos):
    root.mkdir(parents=True,exist_ok=True)
    def status(stage,**data):
        value={'state':'running','stage':stage,**data};atomic_json(root/'progress.json',value);print(json.dumps(value),flush=True)
    lrs2.ROOT=lrs2_downloads.resolve();lrs2.ROOT.mkdir(parents=True,exist_ok=True)
    repo,rev,_=lrs2.SOURCES['train']
    items=[('train',repo,rev,f'data/train-{i:05d}-of-00022.parquet') for i in range(22)]
    items += [(s,r,v,f) for s,(r,v,files) in lrs2.SOURCES.items() if s!='train' for f in files]
    sources=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,record in enumerate(pool.map(lrs2.download,items),1):
            sources.append(record);status('download-lrs2',completed=i,total=len(items))
    atomic_json(root/'lrs2-downloads.json',sources)
    destination=root/'lrs2';videos=destination/'videos';videos.mkdir(parents=True,exist_ok=True)
    # Reuse existing imported clips without storing another copy.
    for p in existing_videos.glob('*.mp4'):
        out=videos/p.name
        if not out.exists():out.hardlink_to(p.resolve())
    report=import_data(root/'lrs2-downloads.json',destination,train_limit=None,validation_limit=None)
    lrs2_rows=json.loads((destination/'manifest.json').read_text())
    for r in lrs2_rows:r['dataset']='LRS2';r['id']='lrs2/'+r['id'];r['format']='rgb-mp4'
    status('lrs2-imported',splits=report['splits'])
    api=f'https://huggingface.co/api/datasets/{LRS3_REPO}'
    meta=requests.get(api,timeout=30);meta.raise_for_status()
    if meta.json().get('gated'):raise ValueError('Public processed LRS3 release now requires access approval')
    base=f'https://huggingface.co/datasets/{LRS3_REPO}/resolve/{LRS3_REV}/'
    for name in ['README.md','shards/bridge-report.json','shards/dataset-manifest.jsonl']:
        response=requests.get(base+name,timeout=60);response.raise_for_status()
        (root/('lrs3-'+Path(name).name)).write_bytes(response.content)
    tree=requests.get(f'{api}/tree/{LRS3_REV}/shards',timeout=30);tree.raise_for_status()
    shards=[s for s in tree.json() if s['path'].endswith('.tar')]
    cropped=root/'lrs3-crops';cropped.mkdir(exist_ok=True)
    records=[];provenance=[]
    for index,shard in enumerate(shards,1):
        ledger=root/(Path(shard['path']).stem+'-import.json')
        if ledger.exists():
            saved=json.loads(ledger.read_text());records.extend(saved['rows']);provenance.append(saved['source']);continue
        if shutil.disk_usage(root).free<12*1024**3:raise ValueError('Less than 12 GiB free; acquisition stopped before filling disk')
        temp=root/'lrs3-download.part';digest=hashlib.sha256();count=0
        with requests.get(base+shard['path'],stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            with temp.open('wb') as f:
                for block in response.iter_content(1024*1024):
                    count+=len(block)
                    if count>1_500_000_000:raise ValueError('Unexpectedly large LRS3 shard')
                    digest.update(block);f.write(block)
        if count!=shard['size']:raise ValueError('LRS3 shard size mismatch')
        expected=shard.get('lfs',{}).get('oid')
        if expected and digest.hexdigest()!=expected:raise ValueError('LRS3 shard checksum mismatch')
        imported=[];pending={}
        with tarfile.open(temp,'r|') as archive:
            for member in archive:
                if not member.isfile():continue
                if member.size>12_000_000:raise ValueError('Unexpected sample size')
                name=Path(member.name).name;key=name.split('.')[0]
                pending.setdefault(key,{})[name[len(key)+1:]]=archive.extractfile(member).read()
                sample=pending[key]
                if not {'roi.npy','txt','json'}<=sample.keys():continue
                info=json.loads(sample['json']);pixels=np.load(io.BytesIO(sample['roi.npy']),allow_pickle=False)
                if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[1:]!=(96,96):raise ValueError('Invalid LRS3 crops')
                if info['fps']!=25 or info['n_frames']!=len(pixels):raise ValueError('Invalid frame metadata')
                text=normalize(sample['txt'].decode())
                identity=str(info['prompt_id']);session=str(info['session_key'])
                # Hold out whole recording sessions using a rule fixed before predictions.
                split='validation' if int(hashlib.sha256(('studio-lrs3-v1:'+session).encode()).hexdigest()[:8],16)%20==0 else 'train'
                output=cropped/(hashlib.sha256(identity.encode()).hexdigest()[:24]+'.npz')
                if not output.exists():np.savez_compressed(output,frames=pixels)
                video_hash=hashlib.sha256(sample['roi.npy']).hexdigest()
                imported.append({'id':'lrs3/'+identity,'split':split,'path':str(output.resolve()),'text':text,'declared_frames':len(pixels),'sha256':video_hash,'dataset':'LRS3-TED','format':'gray-npz','session':session,'source_repository':LRS3_REPO,'source_revision':LRS3_REV})
                del pending[key]
        if pending:raise ValueError('Incomplete sample members')
        source={'repository':LRS3_REPO,'revision':LRS3_REV,'file':shard['path'],'bytes':count,'sha256':digest.hexdigest()}
        atomic_json(ledger,{'source':source,'rows':imported});records.extend(imported);provenance.append(source)
        temp.unlink();status('import-lrs3',completed=index,total=len(shards),clips=len(records))
    combined=lrs2_rows+records;validate_splits(combined)
    sessions={s:{r['session'] for r in records if r['split']==s} for s in ['train','validation']}
    assert not sessions['train']&sessions['validation']
    atomic_json(root/'manifest.json',combined)
    report={'state':'completed','datasets':{d:{s:sum(r['dataset']==d and r['split']==s for r in combined) for s in ['train','validation','test']} for d in ['LRS2','LRS3-TED']},'lrs3_sources':provenance,'lrs2_sources':[{k:v for k,v in x.items() if k!='local_file'} for x in sources],'lrs3_recording_overlap':0,'limitations':['LRS2 main split and LRS3 trainval only; not the full pretraining corpora','LRS3 validation holds out recording sessions from local adaptation, not from original pretraining; speaker independence unverified','Official LRS3 test is not included','Original Auto-AVSR pretrained on both corpora','LRS3 grayscale quantized by distributor; LRS2 stored as RGB video'],'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    atomic_json(root/'provenance.json',report);atomic_json(root/'progress.json',report);print(json.dumps(report['datasets']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--lrs2-downloads',type=Path,required=True);p.add_argument('--existing-videos',type=Path,required=True);a=p.parse_args()
    try:acquire(a.destination.resolve(),a.lrs2_downloads,a.existing_videos)
    except Exception as e:
        atomic_json(a.destination/'progress.json',{'state':'failed','error':f'{type(e).__name__}: {e}'});raise
