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






class VisualFeatures(nn.Module):
    """Input [B,3,T,50,100]; output visual features [B,T,C]."""

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
        return x.permute(0, 2, 1, 3, 4).flatten(2)


