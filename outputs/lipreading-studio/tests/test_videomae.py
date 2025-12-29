import torch
from studio.videomae import CTCHead, TOKENS, greedy, labels, ctc_min_frames
from training.videomae_experiment import metrics, independent_metrics


def test_ctc_blank_separates_repeated_letters():
    a=TOKENS.index('A');b=TOKENS.index('B')
    logits=torch.full((1,7,len(TOKENS)),-10.)
    for i,v in enumerate([a,a,0,a,b,b,0]):logits[0,i,v]=10.
    assert greedy(logits,torch.tensor([7]))==['AAB']
    assert ctc_min_frames(labels('BOOK'))==5


def test_variable_lengths_match_single_sequence():
    torch.set_num_threads(2);torch.manual_seed(1)
    head=CTCHead(hidden=16,layers=2).eval()
    x=torch.randn(2,8,384);lens=torch.tensor([15,9])
    with torch.inference_mode():
        together=head(x,lens)
        for i,n in enumerate(lens.tolist()):
            alone=head(x[i:i+1,:(n+1)//2],torch.tensor([n]))
            assert torch.allclose(together[i,:n],alone[0,:n],atol=1e-6)


def test_training_propagates_finite_gradients():
    head=CTCHead(hidden=16,layers=1)
    x=torch.randn(2,10,384);lengths=torch.tensor([20,17])
    logits=head(x,lengths)
    target=torch.tensor(labels('HELLO')+labels('HI'))
    loss=torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),target,lengths,torch.tensor([5,2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_independent_corpus_metrics_include_empty_predictions():
    rows=[{'dataset':'LRS2','reference':'A CAT','prediction':'A BAT'},
          {'dataset':'LRS3-TED','reference':'HELLO','prediction':''}]
    assert metrics(rows)==independent_metrics(rows)
    assert metrics(rows)['word_errors']==2


def test_gray_corpus_checksum_covers_npy_payload(tmp_path):
    import hashlib
    import numpy as np
    import pytest
    import zipfile
    from training.videomae_experiment import load_crops
    path = tmp_path / 'clip.npz'
    frames = np.zeros((3, 96, 96), dtype=np.uint8)
    np.savez_compressed(path, frames=frames)
    with zipfile.ZipFile(path) as archive:
        digest = hashlib.sha256(archive.read('frames.npy')).hexdigest()
    row = {'id': 'test/clip', 'path': str(path), 'format': 'gray-npz',
           'sha256': digest, 'declared_frames': 3}
    assert np.array_equal(load_crops(row), frames)
    frames[0, 0, 0] = 1
    np.savez_compressed(path, frames=frames)
    with pytest.raises(ValueError, match='NPY payload changed'):
        load_crops(row)
