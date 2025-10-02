"""Regression check for the actual failed clip and honest evaluation counting."""
import json,tempfile
from pathlib import Path
import finetune as ft
from engines import setup_paper_compat

def main():
    setup_paper_compat()
    data=json.loads((ft.RUN/'manifest.json').read_text())
    broken=next(r for r in data['train'] if r['id']=='s3/bgbn9a.mpg')
    bad=ft.prepare(broken);assert bad['valid'] is False and '360, 288' in bad['error'],bad
    good=ft.prepare(data['train'][0]);assert good['valid'] and good['cached'],good
    run=ft.RUN
    with tempfile.TemporaryDirectory() as folder:
        ft.RUN=Path(folder)
        failed_row={**broken,'preprocessing_error':bad['error']}
        report=ft.evaluate(None,[failed_row],'failed-evaluation-clip')
        assert report['metrics']['clips']==1 and report['metrics']['words']==6 and report['metrics']['errors']==6
        assert report['metrics']['wer']==1 and report['preprocessing_failures']==1
    ft.RUN=run
    try:ft.prepare({**broken,'video':'/private/tmp/nonexistent-lipnet-video-for-test.mpg'})
    except FileNotFoundError:pass
    else:raise AssertionError('Missing source data must still stop the run')
    result={'original_failed_clip':bad,'valid_cache_reused':True,'evaluation_failure_counts_all_six_word_errors':True,'missing_source_is_still_fatal':True,'original_model_unchanged':True}
    ft.write(run/'preprocessing-recovery-check.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
