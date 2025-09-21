import torch
import pytest

from lipnet.engine import load_checkpoint, save_checkpoint
from lipnet.data import PreprocessConfig
from lipnet.model import GRUCell, LipNet, ModelConfig, ctc_loss


def test_full_paper_shape_and_convolution_shapes():
    model = LipNet().eval()
    shapes, hooks = [], []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(lambda _, inputs, output: shapes.append(tuple(output.shape))))
    with torch.no_grad():
        result = model(torch.randn(1, 3, 75, 50, 100), [75])
    assert result.shape == (75, 1, 28)
    assert shapes == [(1, 32, 75, 12, 25), (1, 64, 75, 6, 12), (1, 96, 75, 3, 6)]
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("feature_norm", [False, True])
def test_padding_invariance_with_nonzero_biases(feature_norm):
    torch.manual_seed(4)
    model = LipNet(ModelConfig((4, 6, 8), 12, 0.0, feature_norm)).eval()
    for name, parameter in model.named_parameters():
        if "bias" in name:
            torch.nn.init.uniform_(parameter, -0.4, 0.4)
    short = torch.randn(1, 3, 5, 50, 100)
    batch = torch.randn(2, 3, 9, 50, 100)  # deliberately nonzero padding
    batch[0, :, :5] = short[0]
    with torch.no_grad():
        alone = model(short, [5])[:, 0]
        padded = model(batch, [5, 9])[:5, 0]
    torch.testing.assert_close(alone, padded, atol=2e-6, rtol=2e-5)


def test_gru_update_convention():
    cell = GRUCell(1, 1)
    for parameter in cell.parameters():
        torch.nn.init.zeros_(parameter)
    # Zero gates give u=r=0.5, zero candidate gives h_next=0.5*h.
    output = cell(torch.ones(1, 1), torch.tensor([[0.8]]))
    torch.testing.assert_close(output, torch.tensor([[0.4]]))


@pytest.mark.parametrize("feature_norm", [False, True])
def test_full_model_backward_and_checkpoint(tmp_path, feature_norm):
    torch.manual_seed(3)
    model = LipNet(ModelConfig(feature_norm=feature_norm))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    video = torch.randn(1, 3, 5, 50, 100)
    loss = ctc_loss(model(video, [5]), torch.tensor([1, 2]), [5], [2])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0][0].weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, PreprocessConfig(crop=(1, 2, 100, 50)), optimizer, 1, 3.0, {})
    restored, preprocess, state = load_checkpoint(path, "cpu")
    restored.eval()
    assert preprocess.crop == (1, 2, 100, 50)
    assert state["epoch"] == 1
    with torch.no_grad():
        torch.testing.assert_close(model(video, [5]), restored(video, [5]), rtol=0, atol=0)
