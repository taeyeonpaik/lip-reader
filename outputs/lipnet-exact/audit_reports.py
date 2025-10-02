"""Recompute saved prediction metrics independently; verify all source hashes."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def edits(reference,prediction):
    a=reference.split();b=prediction.split();previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def metrics(rows,key='text'):
    e=[edits(r['reference'],r[key]) for r in rows];w=[len(r['reference'].split()) for r in rows]
    rng=np.random.default_rng(9109);idx=rng.integers(0,len(rows),(5000,len(rows)))
    values=np.asarray(e)[idx].sum(1)/np.asarray(w)[idx].sum(1)
    return {'clips':len(rows),'errors':sum(e),'words':sum(w),'wer':sum(e)/sum(w),'exact':sum(r[key]==r['reference'] for r in rows),'conditional_clip_bootstrap_95_interval':np.quantile(values,[.025,.975]).tolist()}

def main():
    manifest=json.loads((ROOT/'reports/source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    reports={}
    for name in ['tutorial','paper']:
        original=json.loads((ROOT/f'reports/{name}-verification.json').read_text());rows=original['predictions'];m=metrics(rows);raw=metrics(rows,'raw')
        for k in ['clips','errors','words','wer','exact']:
            assert m[k]==original['metrics'][k];assert raw[k]==original['raw_metrics'][k]
        speakers=sorted({r['id'].split('/')[0] for r in rows})
        reports[name]={'all':m,'raw':raw,'per_group':{s:metrics([r for r in rows if r['id'].split('/')[0]==s]) for s in speakers}}
        if name=='paper':
            local=metrics([r for r in rows if r['id'].startswith(('s1/','s2/'))]);reports[name]['local_unseen_subset']=local
            assert local['wer']==original['local_unseen_subset_metrics']['wer']
        assert not original['failures']
    result={'source_hashes_verified':len(manifest['sources']),'metrics':reports,'model_selection_or_tuning':False,'uncertainty_limit':'Clip bootstrap is conditional on these particular speakers. One notebook speaker and two paper speakers cannot establish population-level new-speaker accuracy. Tutorial pretraining split unknown; paper split follows upstream declaration. No webcam accuracy claim.'}
    (ROOT/'reports/independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
