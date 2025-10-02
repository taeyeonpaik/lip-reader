"""Reviewed fine-tuned weights with strict integrity checks and author fallback."""
import hashlib,json
from engines import ROOT,UP,Paper
from stable_spelling import stabilize_spelling

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class SelectedPaper(Paper):
    def __init__(self,choice='reviewed',selection_path=None):
        if choice not in ['reviewed','author']:raise ValueError('Unknown checkpoint choice')
        super().__init__('unseen')
        stabilize_spelling(self)
        self.active_kind='author';self.checkpoint_label='Published LipNet unseen-speaker checkpoint';self.selection_error=None
        if choice=='author':return
        path=selection_path or ROOT/'models/paper-selection.json'
        try:
            selection=json.loads(path.read_text())
            if selection['active']=='author':return
            assert selection['format']==1 and selection['active']=='fine_tuned'
            assert selection['architecture']=='rizkiarm-model2-keras' and selection['vocabulary_classes']==28
            assert selection['candidate_path']=='runs/paper-finetune-v1/best.weights.h5'
            assert selection['review_path']=='runs/paper-finetune-v1/independent-review.json'
            assert selection['decoder']=='original-ctc-beam200-spell-alphabetical-ties-v1'
            assert digest(UP/'lipnet/model2.py')==selection['model_source_sha256']
            assert digest(ROOT/'engines.py')==selection['engines_sha256']
            assert digest(ROOT/'stable_spelling.py')==selection['spelling_sha256']
            assert digest(self.weights)==selection['author_sha256']
            candidate=ROOT/selection['candidate_path'];review_path=ROOT/selection['review_path']
            assert digest(candidate)==selection['candidate_sha256'] and digest(review_path)==selection['review_sha256']
            review=json.loads(review_path.read_text())
            assert review['eligible_for_activation'] and review['candidate_sha256']==selection['candidate_sha256']
            assert review['selected_epoch']==selection['selected_epoch'] and review['selected_before_test_verified']
            self.model.load_weights(candidate)  # Strict topological loading; no skipped tensors.
            self.active_kind='fine_tuned';self.checkpoint_label=f"Fine-tuned LipNet, epoch {selection['selected_epoch']}"
        except (OSError,ValueError,KeyError,AssertionError,TypeError) as error:
            # A partially loaded incompatible HDF5 must not leave mixed weights.
            self.lipnet.model.load_weights(str(self.weights))
            self.selection_error=f'{type(error).__name__}: {error}'
    def _predict_input(self,x,start):
        result,p=super()._predict_input(x,start)
        result['mode']=self.checkpoint_label+'; original dlib crop, beam 200, stable spelling ties'
        result['checkpoint']=self.active_kind
        return result,p
