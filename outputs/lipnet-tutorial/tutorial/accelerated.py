"""The unchanged tutorial architecture executed by PyTorch on Apple GPUs.

TensorFlow Metal has no Conv3D kernel. This adapter preserves Keras tensor
ordering, LSTM gate ordering and weights, with explicit parity validation.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

VOCABULARY = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
BLANK = 40


class AcceleratedLipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv3d(1,128,3,padding=1),
                                   nn.Conv3d(128,256,3,padding=1),
                                   nn.Conv3d(256,75,3,padding=1)])
        self.pool = nn.MaxPool3d((1,2,2))
        self.lstm1 = nn.LSTM(6375,128,batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(256,128,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(.5)
        self.classifier = nn.Linear(256,41)
        # Keras has one effective bias per LSTM direction; keep Torch's extra
        # bias fixed at zero so one optimizer step cannot double its update.
        for lstm in (self.lstm1,self.lstm2):
            for suffix in ("", "_reverse"):
                extra = getattr(lstm, "bias_hh_l0"+suffix)
                extra.data.zero_()
                extra.requires_grad_(False)

    def forward(self, frames):
        x = frames.permute(0,4,1,2,3).contiguous()
        for conv in self.convs:
            x = self.pool(torch.relu(conv(x)))
        # Keras TimeDistributed(Flatten) flattens H,W,C, in that order.
        x = x.permute(0,2,3,4,1).contiguous().flatten(2)
        # MPS LSTM does not accept autocast's half input with float weights.
        # Keep recurrent state in float32; the expensive CNN can use float16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.dropout(self.lstm1(x.float())[0])
            x = self.dropout(self.lstm2(x)[0])
        return self.classifier(x)

    @torch.no_grad()
    def load_keras_arrays(self, path):
        with np.load(path) as archive:
            weights = [torch.from_numpy(archive[f"w{i}"]) for i in range(20)]
        index = 0
        for conv in self.convs:
            conv.weight.copy_(weights[index].permute(4,3,0,1,2)); index += 1
            conv.bias.copy_(weights[index]); index += 1
        for lstm in (self.lstm1,self.lstm2):
            for suffix in ("", "_reverse"):
                getattr(lstm,"weight_ih_l0"+suffix).copy_(weights[index].T)
                getattr(lstm,"weight_hh_l0"+suffix).copy_(weights[index+1].T)
                getattr(lstm,"bias_ih_l0"+suffix).copy_(weights[index+2])
                getattr(lstm,"bias_hh_l0"+suffix).zero_()
                index += 3
        self.classifier.weight.copy_(weights[index].T)
        self.classifier.bias.copy_(weights[index+1])

    def save_keras_arrays(self, path):
        weights = []
        for conv in self.convs:
            weights.extend([conv.weight.permute(2,3,4,1,0),conv.bias])
        for lstm in (self.lstm1,self.lstm2):
            for suffix in ("", "_reverse"):
                weights.extend([getattr(lstm,"weight_ih_l0"+suffix).T,
                                getattr(lstm,"weight_hh_l0"+suffix).T,
                                getattr(lstm,"bias_ih_l0"+suffix)+getattr(lstm,"bias_hh_l0"+suffix)])
        weights.extend([self.classifier.weight.T,self.classifier.bias])
        np.savez(path,**{f"w{i}":w.detach().cpu().numpy() for i,w in enumerate(weights)})


def greedy_decode(logits):
    outputs=[]
    for row in logits.detach().argmax(-1).cpu().tolist():
        previous=None
        text=[]
        for token in row:
            if token!=previous and 1<=token<=len(VOCABULARY):
                text.append(VOCABULARY[token-1])
            previous=token
        outputs.append("".join(text))
    return outputs


def loss_ctc(logits, targets, lengths):
    # MPS has no native CTC. The small output tensor crosses to CPU while
    # autograd carries its gradient back to the GPU feature extractor.
    probabilities = logits.float().log_softmax(-1).transpose(0,1).cpu()
    return nn.functional.ctc_loss(probabilities,targets.cpu(),
        torch.full((logits.shape[0],),logits.shape[1],dtype=torch.long),lengths.cpu(),
        blank=BLANK,reduction="sum",zero_infinity=False)/logits.shape[0]
