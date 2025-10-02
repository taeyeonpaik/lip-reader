"""Package sources, published weights, small demos and verified reports."""
from pathlib import Path
import hashlib,json,zipfile
from audit_reports import main as audit
ROOT=Path(__file__).resolve().parent

def main():
    audit()
    compat=json.loads((ROOT/'reports/compatibility-checks.json').read_text())
    runtime=json.loads((ROOT/'reports/runtime-checks.json').read_text())
    ui=json.loads((ROOT/'reports/ui-checks.json').read_text())
    assert compat['historical_ctc_loss_and_decoders_identical'] and compat['scipy_rgb_resize_byte_identical']
    assert all(r['reload_probability_max_error']==0 and r['finite_gradients'] for r in compat['checks'])
    assert runtime['recorded_worker_probability_max_error']==0
    assert ui['browser_upload_prediction_passed'] and ui['random_sentence_generator_passed']
    manifest=json.loads((ROOT/'reports/source-manifest.json').read_text())
    assert hashlib.sha256((ROOT/'models/tutorial-published.npz').read_bytes()).hexdigest()==manifest['published_tutorial_arrays_sha256']
    run=ROOT/'runs/paper-finetune-v1'
    review=json.loads((run/'independent-review.json').read_text());selection=json.loads((ROOT/'models/paper-selection.json').read_text())
    result=json.loads((run/'result.json').read_text());checks=json.loads((ROOT/'reports/model-selection-checks.json').read_text())
    assert result['state']['completed_epoch']==6 and review['selected_before_test_verified']
    assert hashlib.sha256((run/'best.weights.h5').read_bytes()).hexdigest()==review['candidate_sha256']
    assert checks['fine_tuned_checkpoint_probability_max_error']==0 and checks['fallback_probability_max_error']==0
    stable=json.loads((ROOT/'reports/spelling-determinism.json').read_text())
    current_ui=json.loads((ROOT/'reports/ui-finetuning-checks.json').read_text())
    assert stable['same_fixed_policy_across_processes']
    assert current_ui['candidate_upload_passed'] and current_ui['author_upload_passed']
    assert current_ui['reviewed_selector_restored'] and current_ui['results_match_review']
    assert current_ui['random_sentence_generator_passed'] and current_ui['webcam_controls_rendered']
    if selection['active']=='fine_tuned':
        assert review['eligible_for_activation'] and selection['candidate_sha256']==review['candidate_sha256']
        assert selection['review_sha256']==hashlib.sha256((run/'independent-review.json').read_bytes()).hexdigest()
    auto=json.loads((ROOT/'reports/autoavsr-integration.json').read_text())
    auto_ui=json.loads((ROOT/'reports/autoavsr-ui.json').read_text())
    assert auto['recorded_worker_text_matches'] and auto['lipnet_preserved']
    assert auto_ui['autoavsr_upload_passed'] and auto_ui['lipnet_upload_passed'] and auto_ui['autoavsr_camera_controls_rendered']
    dest=ROOT.parent/'lipnet-exact.zip';temp=dest.with_suffix('.tmp')
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(s in p.parts for s in ['__pycache__','.git','.venv','.pytest_cache','resume']) and p.name not in ['.DS_Store','FILE_MANIFEST.json','process.lock'] and p.suffix not in ['.pyc','.tmp']]
    hashes={}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(files):
            data=p.read_bytes();assert len(data)==p.stat().st_size
            name='lipnet-exact/'+str(p.relative_to(ROOT));z.write(p,name);hashes[name]=hashlib.sha256(data).hexdigest()
        z.writestr('lipnet-exact/FILE_MANIFEST.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
        assert not any(n.endswith(('last.pt','best.pt')) or '/work/' in n or '/data/s' in n or '/resume/' in n for n in z.namelist())
        for name in ['README.md','REVIEW.md','COMPATIBILITY.md','PROVENANCE.md','app.py','engines.py','train.py','models/tutorial-published.npz','reports/independent-audit.json','models/paper-selection.json','runs/paper-finetune-v1/best.weights.h5','runs/paper-finetune-v1/independent-review.json','reports/ui-finetuning-checks.json','stable_spelling.py','model_selection.py']:
            assert 'lipnet-exact/'+name in z.namelist()
    temp.replace(dest)
    report={'archive':dest.name,'files':len(hashes)+1,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'zip_crc_and_all_member_sha256_verified':True,'original_published_weights_preserved':True,'fine_tuning_complete':True,'fine_tuning_epochs':6,'selected_epoch':4,'active_checkpoint':selection['active'],'trained_from_scratch':False,'original_full_paper_training_reproduced':False,'optimizer_resumes_included':False,'full_dataset_included':False,'physical_camera_tested':False,'deployed':False,'autoavsr_optional_integration':True,'autoavsr_assets_included':False}
    (ROOT.parent/'LIPNET_EXACT_PACKAGE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
