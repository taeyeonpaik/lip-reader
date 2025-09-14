import itertools
import math
from collections import defaultdict

import pytest
import torch

from lipnet.model import ctc_loss
from lipnet.text import (ErrorRates, collapse, decode, encode, greedy_decode,
                         prefix_beam_search)


def test_repeated_letters_and_blank():
    assert decode(collapse([1, 1, 0, 1, 2, 2, 0])) == "aab"
    assert decode(encode("  Set BLUE by A four please ")) == "set blue by a four please"
    with pytest.raises(ValueError, match="Unsupported"):
        encode("four 4")
    with pytest.raises(ValueError):
        encode("  ")


@pytest.mark.parametrize("seed", range(6))
def test_beam_matches_exhaustive_path_enumeration(seed):
    torch.manual_seed(seed)
    log_probs = torch.randn(4, 3).log_softmax(-1)
    mass = defaultdict(float)
    for path in itertools.product(range(3), repeat=4):
        mass[collapse(path)] += math.exp(sum(float(log_probs[t, token]) for t, token in enumerate(path)))
    expected = decode(max(mass, key=mass.get))
    assert prefix_beam_search(log_probs, 100) == expected


def test_ctc_loss_matches_enumeration_and_has_gradients():
    logits = torch.tensor([[[1., 2., 0.]], [[2., 1., 0.]], [[1., 2., 0.]]], requires_grad=True)
    probs = logits.detach()[:, 0].softmax(-1)
    probability = sum(math.prod(float(probs[t, c]) for t, c in enumerate(path))
                      for path in itertools.product(range(3), repeat=3) if collapse(path) == (1, 1))
    loss = ctc_loss(logits, torch.tensor([1, 1]), [3], [2])
    assert float(loss.detach()) == pytest.approx(-math.log(probability) / 2, abs=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="needs 3 frames"):
        ctc_loss(logits[:2], torch.tensor([1, 1]), [2], [2])


def test_greedy_and_corpus_weighted_metrics():
    assert greedy_decode(torch.tensor([[0., 2.], [0., 3.], [3., 0.], [0., 4.]])) == "aa"
    metric = ErrorRates()
    metric.update("a b c", "a b")
    metric.update("a", "b")
    result = metric.compute()
    assert result["wer"] == 2 / 4
    assert result["cer"] == 3 / 6
    assert result["sentence_accuracy"] == 0
