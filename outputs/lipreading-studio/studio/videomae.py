"""General-video VideoMAE features and a newly trained character CTC lip reader.

The Kinetics-pretrained encoder contains no speech transcription head. Temporal
patch positions are retained; pooling is spatial only. No audio or Auto-AVSR
features/weights enter this model.
"""
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import threading
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vendor.videomae.modeling_pretrain import PretrainVisionTransformerEncoder

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ["<blank>"] + list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
FEATURE_VERSION = "videomae-small-224-dense16-spatialmean-v1"


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sdpa_attention(self, x):
    batch, length, dim = x.shape
    bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
    q, k, v = F.linear(x, self.qkv.weight, bias).reshape(
        batch, length, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return self.proj(y.transpose(1, 2).reshape(batch, length, dim))


def make_encoder(optimized=True):
    model = PretrainVisionTransformerEncoder(embed_dim=384, depth=12, num_heads=6,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.)
    if optimized:
        for block in model.blocks:
            block.attn.forward = types.MethodType(sdpa_attention, block.attn)
    return model


class VideoFeatures:
    def __init__(self, device="auto", verify=True):
        torch.set_num_threads(4)
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device
        self.path = ROOT / 'models/videomae-small-kinetics.pt'
        self.provenance = json.loads((ROOT / 'models/videomae-provenance.json').read_text())
        if verify and sha256(self.path) != self.provenance['encoder_sha256']:
            raise ValueError('VideoMAE encoder checksum mismatch')
        self.model = make_encoder()
        self.model.load_state_dict(torch.load(self.path, map_location='cpu', weights_only=True), strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None, None]

    @torch.inference_mode()
    def __call__(self, crops):
        if crops.dtype != np.uint8 or crops.ndim not in (3, 4) or not len(crops):
            raise ValueError('Expected nonempty uint8 mouth crops')
        if crops.shape[1:3] != (96, 96):
            raise ValueError('Expected 96x96 mouth crops')
        if crops.ndim == 3:
            crops = np.repeat(crops[..., None], 3, axis=-1)
        if crops.shape[-1] != 3:
            raise ValueError('Expected RGB channels')
        outputs = []
        for start in range(0, len(crops), 16):
            frames = crops[start:start+16]
            valid = len(frames)
            if valid < 16:
                frames = np.concatenate([frames, np.repeat(frames[-1:], 16-valid, axis=0)])
            x = torch.from_numpy(frames.copy()).to(self.device).permute(0, 3, 1, 2).float() / 255.
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = (x.permute(1, 0, 2, 3).unsqueeze(0) - self.mean) / self.std
            mask = torch.zeros((1, 8*14*14), dtype=torch.bool, device=self.device)
            features = self.model(x, mask).reshape(1, 8, 196, 384).mean(dim=2)[0]
            # One feature per two input frames. Keep the final real partial pair.
            outputs.append(features[:math.ceil(valid/2)].cpu())
        return torch.cat(outputs)


class CTCHead(nn.Module):
    def __init__(self, hidden=256, layers=3, dropout=.15):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, dropout=dropout)
        self.input = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, layers, batch_first=True,
                               bidirectional=True, dropout=dropout if layers > 1 else 0.)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(2*hidden, len(TOKENS)))
        # Avoid starting with a strong CTC blank preference.
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, lengths):
        # Restore 25 Hz alignment slots after VideoMAE's two-frame tubelets.
        # This adds no visual information; the GRU learns temporal context.
        x = self.input(features.repeat_interleave(2, dim=1))
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.temporal(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self.output(x)


def labels(text):
    from training.metrics import normalize
    mapping = {v: i for i, v in enumerate(TOKENS)}
    return [mapping[c] for c in normalize(text)]


def ctc_min_frames(target):
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def greedy(logits, lengths):
    result = []
    for sequence, length in zip(logits.argmax(-1).tolist(), lengths.tolist()):
        previous = -1
        out = []
        for token in sequence[:length]:
            if token and token != previous:
                out.append(TOKENS[token])
            previous = token
        result.append(' '.join(''.join(out).split()))
    return result


class VideoMAEReader:
    def __init__(self, checkpoint=None, device='auto'):
        self.features = VideoFeatures(device)
        self.device = self.features.device
        self.lock = threading.Lock()
        self.beam_size = 1
        self.decoding = {'method': 'CTC greedy', 'beam_size': 1, 'lm_weight': 0.0}
        self.model_origin = 'VideoMAE Small Kinetics encoder + locally trained CTC head (experimental)'
        path = Path(checkpoint or ROOT / 'models/videomae-ctc.pt')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') != 'videomae-ctc-v1' or saved.get('tokens') != TOKENS:
            raise ValueError('Incompatible VideoMAE transcription checkpoint')
        if saved.get('feature_version') != FEATURE_VERSION or saved.get('encoder_sha256') != self.features.provenance['encoder_sha256']:
            raise ValueError('VideoMAE checkpoint uses a different feature encoder')
        self.head = CTCHead(**saved['head_config'])
        self.head.load_state_dict(saved['state_dict'], strict=True)
        self.head.eval()

    def recognize(self, crops):
        with self.lock, torch.inference_mode():
            features = self.features(crops).float()
            lengths = torch.tensor([len(crops)])
            return greedy(self.head(features.unsqueeze(0), lengths), lengths)[0]

    def transcribe(self, normalized_video, on_segment=None):
        # Reuse only face tracking, segmentation and result formatting.
        from studio.pipeline import LipReader
        return LipReader.transcribe(self, normalized_video, on_segment)
