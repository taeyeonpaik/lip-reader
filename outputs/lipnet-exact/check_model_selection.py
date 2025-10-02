"""Verify activation, exact checkpoint loading and fail-closed author fallback."""
import json,tempfile
from pathlib import Path
import numpy as np
from engines import ROOT,UP,Paper
from model_selection import SelectedPaper

def main():
    selection=json.loads((ROOT/'models/paper-selection.json').read_text())
    original=Paper();x=original.prepare(UP/'evaluation/samples/bbaf2n',mouth=True)
    author=original.model(x,training=False).numpy()
    original.model.load_weights(ROOT/selection['candidate_path']);expected=original.model(x,training=False).numpy()
    actual=SelectedPaper();assert actual.active_kind=='fine_tuned' and actual.selection_error is None
    assert np.array_equal(actual.model(x,training=False),expected)
    explicit=SelectedPaper('author');assert explicit.active_kind=='author' and np.array_equal(explicit.model(x,training=False),author)
    failures=[]
    for key,value in [('format',999),('candidate_sha256','0'*64),('review_sha256','0'*64),('candidate_path','../outside.h5'),('vocabulary_classes',41),('decoder','different-decoder')]:
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'selection.json';p.write_text(json.dumps({**selection,key:value}))
            fallback=SelectedPaper(selection_path=p)
            assert fallback.active_kind=='author' and fallback.selection_error
            assert np.array_equal(fallback.model(x,training=False),author)
            failures.append(key)
    report={'fine_tuned_checkpoint_probability_max_error':0.0,'explicit_author_probability_max_error':0.0,'invalid_selection_fields_fall_back':failures,'fallback_probability_max_error':0.0,'author_neural_weights_unchanged':True,'decoder':'original beam200 scores with stable alphabetical spelling ties'}
    (ROOT/'reports/model-selection-checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
