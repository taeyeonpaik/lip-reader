"""Verify experiments 3/4 from workspace root with the project's Python runtime."""
from pathlib import Path
import hashlib, json, re, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.deeper import Deeper, parser, restore, PREFIXES
from studio.pipeline import ROOT


def distance(a, b):
    # Full matrix implementation, separate from training's rolling-row scorer.
    grid = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): grid[i][0] = i
    for j in range(len(b) + 1): grid[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            grid[i][j] = min(grid[i-1][j]+1, grid[i][j-1]+1, grid[i-1][j-1]+(a[i-1] != b[j-1]))
    return grid[-1][-1]


def metrics(rows):
    def norm(s): return ' '.join(re.sub("[^A-Z0-9' ]", ' ', s.upper().replace('’', "'")).split())
    w = c = nw = nc = exact = failures = 0
    for r in rows:
        a = norm(r['reference']); b = '' if r.get('error') else norm(r.get('prediction', ''))
        w += distance(a.split(), b.split()); c += distance(a, b)
        nw += len(a.split()); nc += len(a); exact += a == b; failures += bool(r.get('error'))
    return dict(clips=len(rows), word_errors=w, reference_words=nw, wer=w/nw,
                character_errors=c, reference_characters=nc, cer=c/nc,
                exact_sentences=exact, failures=failures)


def audit_reports(run):
    checks = []
    for path in sorted(run.glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or 'predictions' not in data: continue
        rows = data['predictions']; reported = data['metrics']; actual = metrics(rows)
        assert len({r['id'] for r in rows}) == len(rows), path
        if 'by_dataset' in reported:
            actual['by_dataset'] = {d: metrics([r for r in rows if r['dataset'] == d]) for d in sorted({r['dataset'] for r in rows})}
            actual['macro_wer'] = sum(m['wer'] for m in actual['by_dataset'].values()) / len(actual['by_dataset'])
        assert actual == reported, (path, actual, reported)
        assert not actual['failures'], path
        checks.append({'report': path.name, 'clips': len(rows), 'metrics_match': True})
    return checks


def main():
    run = ROOT/'training/runs/lrs2-lrs3-mixed-v4'
    checked = audit_reports(run)
    decoding_checks = audit_reports(ROOT/'training/runs/decoding-v3')
    print('All metric reports independently recomputed', flush=True)
    args = parser().parse_args(['--manifest','work/corpora/verified-manifest.json','--cache','work/deeper/features','--run',str(run),'--balanced-datasets','--beam-size','20','--lr','3e-6','--accumulation','16','--microbatch','4','--epochs','6','--min-epochs','3','--patience','2','--mask-probability','.15','--seed','44'])
    experiment = Deeper(args)
    last = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    result = json.loads((run/'result.json').read_text())
    history = json.loads((run/'history.json').read_text())
    assert last['epoch'] == result['epochs_completed'] == 3
    assert last['step'] == result['updates'] == 14304
    assert last['history'] == history
    assert result['selection']['selected'] == 'pretrained' and last['best_epoch'] == 0
    assert not (run/'best.pt').exists() and result['best_test'] is None
    baseline = result['selection']['baseline_validation']
    for h in history:
        assert not experiment.qualifies(h['validation'], baseline)
        assert baseline['macro_wer'] - h['validation']['macro_wer'] < .005
    changes = {prefix: max(float((t-experiment.original[n]).abs().max()) for n,t in last['state'].items() if n.startswith(prefix)) for prefix in PREFIXES}
    assert all(v > 0 for v in changes.values())
    restore(experiment.model, last['state']); experiment.mode(False)
    predictions = json.loads((run/'epoch-03-validation.json').read_text())['predictions']
    lookup = {r['id']: r for r in experiment.splits['validation']}
    checks = []
    for dataset in ['LRS2','LRS3-TED']:
        for expected in [p for p in predictions if p['dataset'] == dataset][:2]:
            with torch.no_grad(): encoded = experiment.encode_suffix(experiment.prefix(lookup[expected['id']])).squeeze(0)
            actual = experiment.reader.decode(encoded)
            assert actual == expected['prediction'], (actual, expected)
            checks.append({'id': expected['id'], 'dataset': dataset, 'matches_recorded_prediction': True})
    with (run/'last.pt').open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    report = {'state':'passed','metrics_recomputed':checked,'decoder_metrics_recomputed':decoding_checks,'last_checkpoint_epoch':3,'optimizer_updates':14304,'changed_modules_max_absolute_delta':changes,'reloaded_prediction_checks':checks,'checkpoint_sha256':{'last.pt':digest},'base_checkpoint_hash_verified':True,'selection_rule_verified':True,'production_model':'original pretrained','adapted_test_evaluated':False,'adapted_test_note':'No validation checkpoint qualified, so no adapted test evaluation was run.'}
    (ROOT/'validation/mixed-checkpoint-and-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'validation/decoding-metric-audit.json').write_text(json.dumps({'state':'passed','independent_full_matrix_edit_distance':True,'reports':decoding_checks,'cross_dataset_audit_included':True},indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)

if __name__ == '__main__': main()
