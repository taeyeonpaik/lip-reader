"""LipNet architecture, including a GRU implementing the paper's equations."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .text import NUM_CLASSES


@dataclass
class ModelConfig:
    channels: tuple = (32, 64, 96)
    hidden_size: int = 256
    dropout: float = 0.5
    # Optional practical stabilization; disabled for the original paper model.
    feature_norm: bool = False

    def to_dict(self):
        return asdict(self)


class GRUCell(nn.Module):
    """Reset-before-matrix-multiply GRU, as in LipNet section 3.2.

    u,r = sigmoid(Wx + Wh + b); n = tanh(Ux + U(r*h) + b)
    h_next = (1-u)*h + u*n. PyTorch's built-in GRU uses a different reset
    placement, so we explicitly implement the recurrence here.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gates = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.input_candidate = nn.Linear(input_size, hidden_size)
        self.hidden_candidate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        update, reset = torch.sigmoid(self.input_gates(x) + self.hidden_gates(h)).chunk(2, -1)
        candidate = torch.tanh(self.input_candidate(x) + self.hidden_candidate(reset * h))
        return (1 - update) * h + update * candidate


class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_cell = GRUCell(input_size, hidden_size)
        self.backward_cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, lengths):
        batch, time, _ = x.shape
        directions = []
        for cell, indices in [(self.forward_cell, range(time)),
                              (self.backward_cell, range(time - 1, -1, -1))]:
            h = x.new_zeros(batch, self.hidden_size)
            outputs = [None] * time
            for t in indices:
                valid = (t < lengths).unsqueeze(-1)
                h = torch.where(valid, cell(x[:, t], h), h)
                outputs[t] = torch.where(valid, h, torch.zeros_like(h))
            directions.append(torch.stack(outputs, dim=1))
        return torch.cat(directions, dim=-1)


class LipNet(nn.Module):
    """Input [B,3,T,50,100]; output unnormalised logits [T,B,28]."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        if len(c) != 3 or any(v < 1 for v in c) or self.config.hidden_size < 1:
            raise ValueError("Three positive convolution widths and a positive hidden size required")
        self.blocks = nn.ModuleList()
        for cin, cout, kernel, stride in [(3, c[0], (3, 5, 5), (1, 2, 2)),
                                          (c[0], c[1], (3, 5, 5), (1, 1, 1)),
                                          (c[1], c[2], (3, 3, 3), (1, 1, 1))]:
            self.blocks.append(nn.Sequential(
                nn.Conv3d(cin, cout, kernel, stride, tuple(k // 2 for k in kernel)),
                nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.Dropout3d(self.config.dropout)))
        self.gru1 = BidirectionalGRU(c[2] * 3 * 6, self.config.hidden_size)
        self.gru2 = BidirectionalGRU(2 * self.config.hidden_size, self.config.hidden_size)
        self.feature_norm = nn.LayerNorm(c[2] * 3 * 6) if self.config.feature_norm else nn.Identity()
        self.classifier = nn.Linear(2 * self.config.hidden_size, NUM_CLASSES)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, GRUCell):
                if self.config.feature_norm:
                    nn.init.xavier_uniform_(module.input_gates.weight)
                    nn.init.xavier_uniform_(module.input_candidate.weight)
                for square in module.hidden_gates.weight.chunk(2, 0):
                    nn.init.orthogonal_(square)
                nn.init.orthogonal_(module.hidden_candidate.weight)

    def forward(self, video, lengths):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[3:] != (50, 100):
            raise ValueError("Expected video shape [batch, 3, time, 50, 100]")
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=video.device)
        if lengths.shape != (video.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > video.shape[2]):
            raise ValueError("Each sequence length must be between 1 and the padded time dimension")
        mask = (torch.arange(video.shape[2], device=video.device)[None, :] < lengths[:, None])
        mask = mask[:, None, :, None, None]
        # Remask each convolutional block: biases and temporal convolutions must
        # not let padding flow back into the valid end of a shorter video.
        x = video * mask
        for block in self.blocks:
            x = block(x) * mask
        x = self.feature_norm(x.permute(0, 2, 1, 3, 4).flatten(2))
        x = self.gru2(self.gru1(x, lengths), lengths)
        return self.classifier(x).transpose(0, 1)


def ctc_loss(logits, targets, input_lengths, target_lengths):
    """Validate alignability, including repeated labels, before PyTorch CTC."""
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long, device="cpu")
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long, device="cpu")
    if input_lengths.shape != (logits.shape[1],) or target_lengths.shape != input_lengths.shape:
        raise ValueError("Length vectors must match batch size")
    if torch.any(input_lengths < 1) or torch.any(input_lengths > logits.shape[0]):
        raise ValueError("Invalid input lengths")
    if torch.any(target_lengths < 1) or int(target_lengths.sum()) != targets.numel():
        raise ValueError("Invalid concatenated targets")
    if torch.any(targets <= 0) or torch.any(targets >= NUM_CLASSES):
        raise ValueError("Targets must be non-blank vocabulary IDs")
    offset = 0
    for length, available in zip(target_lengths.tolist(), input_lengths.tolist()):
        labels = targets[offset:offset + length]
        required = length + int((labels[1:] == labels[:-1]).sum())
        if available < required:
            raise ValueError(f"CTC needs {required} frames for this transcript, got {available}")
        offset += length
    return nn.functional.ctc_loss(logits.log_softmax(-1), targets, input_lengths,
                                  target_lengths, blank=0, reduction="mean", zero_infinity=False)
