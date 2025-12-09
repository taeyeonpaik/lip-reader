"""Evaluate a frozen decoder choice on extra LRS2 validation and LRS3 sessions."""
import argparse,hashlib,json
from pathlib import Path
from training.decoding_experiment import DecoderExperiment
from training.data import atomic_json,validate_splits

def audit(args):
    root=args.run
    completed=json.loads((root/'result.json').read_text())
    selected=completed['selection']['candidate']
    combined=json.loads(args.combined_manifest.read_text());validate_splits(combined)
    engine=DecoderExperiment(args)
    original={'lrs2/'+r['id'] for r in engine.rows if r['split']=='validation'}
    groups={'lrs2-extra-validation':[r for r in combined if r['dataset']=='LRS2' and r['split']=='validation' and r['id'] not in original],
            'lrs3-session-validation':sorted([r for r in combined if r['dataset']=='LRS3-TED' and r['split']=='validation'],key=lambda r:hashlib.sha256(('decoder-audit:'+r['id']).encode()).hexdigest())[:500]}
    if any(not v for v in groups.values()):raise ValueError('Missing cross-dataset audit clips')
    output={}
    atomic_json(root/'cross-dataset-plan.json',{'selected_before_audit':selected,'groups':{k:[r['id'] for r in rows] for k,rows in groups.items()},'combined_manifest_sha256':hashlib.sha256(args.combined_manifest.read_bytes()).hexdigest(),'selection_changes_allowed':False})
    for group,rows in groups.items():
        baseline=engine.evaluate(rows,'baseline',group+'-baseline')
        candidate=baseline if selected=='baseline' else engine.evaluate(rows,selected,group+'-selected')
        output[group]={'baseline':baseline['metrics'],'selected':candidate['metrics'],'baseline_mean_seconds':baseline['mean_seconds'],'selected_mean_seconds':candidate['mean_seconds']}
    atomic_json(root/'cross-dataset-result.json',{'state':'completed','selected_before_audit':selected,'groups':output,'note':'LRS3 samples are held out of local adaptation by session; the original pretrained model already used LRS3. This is not the official LRS3 test benchmark.'})
    engine.status('cross-dataset-complete',state='completed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','combined-manifest','cache','run']:p.add_argument('--'+key,type=Path,required=True)
    audit(p.parse_args())
