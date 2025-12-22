"""Verify the completed bounded MWER run; never train or select a new epoch."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from training.mwer import MWERExperiment,parser,restore,PREFIXES,qualifies
from training.metrics import normalize,edit_distance
from validation.check_mixed_completion import metrics as independent_metrics
from studio.pipeline import ROOT


def recompute(rows):
    result=independent_metrics(rows)
    result['by_dataset']={d:independent_metrics([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})}
    result['macro_wer']=sum(m['wer'] for m in result['by_dataset'].values())/len(result['by_dataset'])
    return result


def paired_bootstrap(baseline,candidate,repetitions=2000):
    """Resample recording IDs within each dataset, preserving paired predictions."""
    a={r['id']:r for r in baseline};b={r['id']:r for r in candidate}
    assert set(a)==set(b)
    grouped={}
    for key,left in a.items():
        right=b[key];assert left['reference']==right['reference'] and left['dataset']==right['dataset']
        ref=normalize(left['reference']).split();session=key.rsplit('/',1)[0];d=left['dataset']
        values=grouped.setdefault(d,{}).setdefault(session,np.zeros(3))
        values+=np.array([edit_distance(ref,normalize(left['prediction']).split()),edit_distance(ref,normalize(right['prediction']).split()),len(ref)])
    arrays={d:np.stack(list(sessions.values())) for d,sessions in grouped.items()}
    rng=np.random.default_rng(452000);deltas=[]
    for _ in range(repetitions):
        sampled=[]
        for arr in arrays.values():
            values=arr[rng.integers(0,len(arr),len(arr))].sum(0);sampled.append((values[0]-values[1])/values[2])
        deltas.append(np.mean(sampled))
    point=recompute(baseline)['macro_wer']-recompute(candidate)['macro_wer']
    lo,hi=np.quantile(deltas,[.025,.975])
    return {'method':'paired recording-session bootstrap, stratified by dataset','replicates':repetitions,'seed':452000,'recording_groups':{d:len(s) for d,s in grouped.items()},'macro_wer_gain':point,'gain_95_percent_interval':[float(lo),float(hi)],'interval_excludes_zero_in_favor_of_candidate':bool(lo>0),'note':'Validation-selected comparisons are exploratory; intervals do not remove selection bias or original-pretraining overlap.'}


def main():
    run=ROOT/'training/runs/mwer-v5';result=json.loads((run/'result.json').read_text())
    assert result['state']=='completed'
    checked=[]
    for path in sorted(run.glob('*.json')):
        data=json.loads(path.read_text())
        if isinstance(data,dict) and 'predictions' in data:
            assert recompute(data['predictions'])==data['metrics'],path
            assert not data['metrics']['failures'],path
            checked.append(path.name)
    lists=[json.loads(l) for l in (run/'diagnostic-nbest.jsonl').read_text().splitlines()]
    oracle=[{'id':r['id'],'dataset':r['dataset'],'reference':r['reference'],'prediction':min(r['candidates'],key=lambda c:edit_distance(normalize(r['reference']).split(),normalize(c['text']).split()))['text']} for r in lists]
    diagnosis=json.loads((run/'diagnostic.json').read_text())
    assert recompute(oracle)==diagnosis['oracle']
    e=MWERExperiment(parser().parse_args([]));reloads=[];hashes={};changes={};uncertainty={}
    baseline=json.loads((run/'baseline-validation.json').read_text())
    lookup={r['id']:r for r in e.groups['validation']}
    if result['training_started']:
        assert diagnosis['gate_passed']
        best_by_arm={}
        for arm in ['ce','mwer']:
            history=json.loads((run/(arm+'-history.json')).read_text());assert len(history)==2
            last=torch.load(run/(arm+'-last.pt'),map_location='cpu',weights_only=True)
            assert last['history']==history and last['step']==250 and last['epoch']==2
            best=min(history,key=lambda h:h['validation']['macro_wer'])
            best_by_arm[arm]=json.loads((run/f"{arm}-epoch-{best['epoch']:02}-validation.json").read_text())
            uncertainty['baseline_vs_'+arm]=paired_bootstrap(baseline['predictions'],best_by_arm[arm]['predictions'])
            for epoch in [1,2]:
                path=run/f'{arm}-epoch-{epoch:02}.pt';bundle=torch.load(path,map_location='cpu',weights_only=True)
                assert bundle['format']==3 and bundle['base_sha256']==e.config['base_sha256'] and bundle['prefixes']==list(PREFIXES)
                report=json.loads((run/f'{arm}-epoch-{epoch:02}-validation.json').read_text());assert bundle['validation']==report['metrics']==history[epoch-1]['validation']
                changes[path.name]={prefix:max(float((t-e.original[n]).abs().max()) for n,t in bundle['state'].items() if n.startswith(prefix)) for prefix in PREFIXES}
                assert all(delta>0 for delta in changes[path.name].values())
                restore(e.model,bundle['state']);e.model.eval()
                for d in ['LRS2','LRS3-TED']:
                    expected=next(r for r in report['predictions'] if r['dataset']==d)
                    actual=e.reader.decode(e.features(lookup[expected['id']]))
                    assert actual==expected['prediction'],(path.name,expected['id'],actual,expected['prediction'])
                    reloads.append({'checkpoint':path.name,'id':expected['id'],'matches':True})
                with path.open('rb') as f:hashes[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
        uncertainty['ce_vs_mwer']=paired_bootstrap(best_by_arm['ce']['predictions'],best_by_arm['mwer']['predictions'])
        eligible=[h for h in result['selection']['candidates'] if qualifies(h['validation'],baseline['metrics'])]
        winner=min(eligible,key=lambda h:h['validation']['macro_wer']) if eligible else None
        assert result['selection']['selected']==({'arm':winner['arm'],'epoch':winner['epoch']} if winner else 'pretrained')
        for suffix in ['audit','test']:
            if (run/f'selected-{suffix}.json').exists():
                left=json.loads((run/f'baseline-{suffix}.json').read_text());right=json.loads((run/f'selected-{suffix}.json').read_text())
                uncertainty['selected_'+suffix]=paired_bootstrap(left['predictions'],right['predictions'])
    report={'state':'passed','metrics_recomputed':checked,'oracle_recomputed':True,'base_hash_verified':True,'checkpoint_sha256':hashes,'changed_modules_max_abs_delta':changes,'reloaded_predictions':reloads,'selection_verified':True,'uncertainty':uncertainty,'production_changed':False}
    (ROOT/'validation/mwer-checkpoint-and-metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
