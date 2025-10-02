"""Verify one fixed tie policy across different Python hash randomizations."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def worker():
    from types import SimpleNamespace
    from engines import setup_paper_compat,UP
    setup_paper_compat()
    from lipnet.utils.spell import Spell
    from stable_spelling import stabilize_spelling
    model=stabilize_spelling(SimpleNamespace(spell=Spell(str(UP/'common/dictionaries/grid.txt'))))
    values=[]
    for name in ['baseline-validation','selected-reload-validation','baseline-test','candidate-test']:
        d=json.loads((ROOT/'runs/paper-finetune-v1'/(name+'.json')).read_text())
        values.extend(model.spell.sentence(r['raw']) for r in d['predictions'])
    print(json.dumps({'predictions':len(values),'sha256':hashlib.sha256(json.dumps(values).encode()).hexdigest()}),flush=True)

if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:
        results=[]
        for seed in ['0','917']:
            env={**os.environ,'PYTHONHASHSEED':seed}
            out=subprocess.check_output([sys.executable,__file__,'--worker'],env=env,text=True)
            results.append({'python_hash_seed':seed,**json.loads(out.splitlines()[-1])})
        assert results[0]['sha256']==results[1]['sha256']
        report={'same_fixed_policy_across_processes':True,'policy':'alphabetical order among equal original spelling probabilities','reference_text_supplied':False,'runs':results}
        (ROOT/'reports/spelling-determinism.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
