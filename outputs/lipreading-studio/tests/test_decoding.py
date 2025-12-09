import torch
import pytest
from studio.pipeline import LipReader  # initializes the vendored ESPnet import path
from studio.language_model import EnglishTransformerLM


def tiny_lm():
    torch.manual_seed(8)
    return EnglishTransformerLM({'model_module':'transformer','pos_enc':'none','tie_weights':False,'att_unit':16,'embed_unit':8,'n_vocab':31,'head':2,'unit':32,'layer':2}).eval()


def test_lm_cached_beams_match_full_prefix_after_branching():
    lm=tiny_lm();tokens=torch.tensor([[30,4,9,2],[30,4,7,8],[30,6,5,3]])
    with torch.inference_mode():
        _,states=lm.batch_score(tokens[:,:2],[None]*3,torch.empty(3,1,1))
        # Select, duplicate and reorder predecessor states as beam search does.
        order=[2,0,0];branched=tokens[order,:3].clone();branched[2,-1]=11
        scores,_=lm.batch_score(branched,[states[i] for i in order],torch.empty(3,1,1))
        reference=lm.logits(branched)[0][:,-1].log_softmax(-1)
        torch.testing.assert_close(scores,reference,atol=2e-6,rtol=1e-5)
        assert torch.isfinite(scores).all()
        torch.testing.assert_close(scores.exp().sum(-1),torch.ones(3))


def test_lm_is_causal_and_score_api_matches_batch():
    lm=tiny_lm();a=torch.tensor([[30,4,9,2]]);b=torch.tensor([[30,4,9,20]])
    with torch.inference_mode():
        torch.testing.assert_close(lm.logits(a)[0][:,:3],lm.logits(b)[0][:,:3],atol=0,rtol=0)
        score,_=lm.score(a[0],None,torch.empty(1,1))
        batch,_=lm.batch_score(a,[None],torch.empty(1,1,1))
        torch.testing.assert_close(score,batch[0])


@pytest.mark.parametrize('kwargs',[{'beam_size':0},{'lm_weight':.3},{'ctc_weight':float('nan')},{'lm_weight':-1}])
def test_invalid_decoder_settings_rejected_before_loading_model(kwargs):
    reader=LipReader.__new__(LipReader)
    with pytest.raises(ValueError):reader.configure_decoding(**kwargs)


def test_lrs3_grayscale_inputs_are_not_converted_twice(tmp_path):
    import numpy as np
    from training.inputs import load_pixels
    frames=np.full((3,96,96),127,dtype=np.uint8)
    path=tmp_path/'crop.npz';np.savez_compressed(path,frames=frames)
    pixels=load_pixels({'format':'gray-npz','path':str(path),'declared_frames':3},'cpu')
    assert pixels.shape==(1,3,1,88,88)
    torch.testing.assert_close(pixels,torch.full_like(pixels,(127/255-.421)/.165))
    with pytest.raises(ValueError,match='Invalid grayscale'):
        load_pixels({'format':'gray-npz','path':str(path),'declared_frames':4},'cpu')


def test_invalid_grayscale_dtype_rejected(tmp_path):
    import numpy as np
    from training.inputs import load_pixels
    path=tmp_path/'invalid.npz';np.savez_compressed(path,frames=np.zeros((3,96,96),dtype=np.float32))
    with pytest.raises(ValueError,match='Invalid grayscale'):
        load_pixels({'format':'gray-npz','path':str(path),'declared_frames':3},'cpu')
