import copy

import pytest
import torch

from training.metrics import edit_distance, normalize, summarize
from training.data import validate_splits
from studio.adaptation import head_state, set_heads, save_adapter, load_adapter


def test_corpus_errors_include_failed_clips_and_insertions():
    result = summarize([
        {"reference": "ONE TWO THREE", "prediction": "ONE FOUR THREE FIVE"},
        {"reference": "SIX SEVEN", "prediction": "should be ignored", "error": "decode failed"},
    ])
    assert result["reference_words"] == 5
    assert result["word_errors"] == 4
    assert result["wer"] == .8
    assert result["failures"] == 1
    assert normalize("Hello,  it’s me!") == "HELLO IT'S ME"
    assert edit_distance("kitten", "sitting") == 3


def test_split_validation_rejects_id_and_content_leakage():
    a = {"id": "a", "split": "train", "sha256": "hash", "text": "A WORD"}
    b = {"id": "b", "split": "test", "sha256": "hash", "text": "A WORD"}
    with pytest.raises(ValueError, match="Identical video"):
        validate_splits([a, b])
    with pytest.raises(ValueError, match="Duplicate sample"):
        validate_splits([a, dict(a)])


def model():
    m = torch.nn.Module()
    m.ctc = torch.nn.Module(); m.ctc.ctc_lo = torch.nn.Linear(3,4)
    m.decoder = torch.nn.Module(); m.decoder.output_layer = torch.nn.Linear(3,4)
    m.frozen_encoder = torch.nn.Linear(3,3)
    return m


def test_adapter_roundtrip_does_not_change_encoder_and_rejects_invalid_weights(tmp_path):
    a, b = model(), model()
    original_encoder = copy.deepcopy(b.frozen_encoder.state_dict())
    save_adapter(a, tmp_path/"adapter.pt", {"epoch": 3})
    assert load_adapter(b,tmp_path/"adapter.pt") == {"epoch":3}
    for key, tensor in head_state(a).items():
        torch.testing.assert_close(tensor,head_state(b)[key])
    for key,tensor in original_encoder.items():
        torch.testing.assert_close(tensor,b.frozen_encoder.state_dict()[key])
    invalid = head_state(a)
    invalid["ctc.ctc_lo.weight"][0,0] = float("nan")
    with pytest.raises(ValueError, match="Invalid adapter"):
        set_heads(b,invalid)
