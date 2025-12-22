import torch
import pytest
from training.mwer import expected_word_error, qualifies


def test_mwer_gradient_rewards_lower_word_errors():
    scores=torch.tensor([0.,0.],requires_grad=True)
    errors=torch.tensor([0.,3.])
    loss=expected_word_error(scores,errors);loss.backward()
    assert scores.grad[0]<0 and scores.grad[1]>0
    updated=scores.detach()-.1*scores.grad
    assert (updated.softmax(0)*errors).sum()<(scores.detach().softmax(0)*errors).sum()


def test_centering_preserves_gradient_and_equal_errors_have_zero_gradient():
    scores=torch.tensor([-.4,.2,-1.],requires_grad=True);errors=torch.tensor([1.,4.,2.])
    a=torch.autograd.grad(expected_word_error(scores,errors),scores,retain_graph=True)[0]
    b=torch.autograd.grad((scores.softmax(0)*errors).sum(),scores)[0]
    assert torch.allclose(a,b,atol=1e-6)
    equal=torch.ones(3)
    assert torch.equal(torch.autograd.grad(expected_word_error(scores,equal),scores)[0],torch.zeros(3))
    with pytest.raises(ValueError):expected_word_error(torch.tensor([float('nan')]),torch.tensor([1.]))


def test_mwer_selection_requires_mean_gain_and_limits_dataset_regression():
    base={'failures':0,'macro_wer':.2,'cer':.1,'by_dataset':{'a':{'wer':.3},'b':{'wer':.1}}}
    candidate={'failures':0,'macro_wer':.19,'cer':.09,'by_dataset':{'a':{'wer':.301},'b':{'wer':.079}}}
    assert qualifies(candidate,base)
    candidate['by_dataset']['a']['wer']=.303
    assert not qualifies(candidate,base)
