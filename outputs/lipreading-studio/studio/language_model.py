"""Inference-only ESPnet Transformer LM compatible with the author's English weights.

Uses the Apache-2.0 ESPnet layer equations and checkpoint names. Restricted to
this release's pre-normalized, position-free Transformer; no training imports.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class LMLayer(nn.Module):
    def __init__(self,width,heads,units):
        super().__init__()
        self.self_attn=MultiHeadedAttention(heads,width,0.)
        self.feed_forward=PositionwiseFeedForward(width,units,0.)
        self.norm_ff=LayerNorm(width);self.norm_mha=LayerNorm(width)

    def forward(self,x,mask,cache=None):
        residual=x;normalized=self.norm_mha(x)
        query=normalized
        if cache is not None:
            if cache.shape!=(x.shape[0],x.shape[1]-1,x.shape[2]):raise ValueError('Invalid LM cache shape')
            query=query[:,-1:];residual=residual[:,-1:];mask=mask[:,-1:]
        x=residual+self.self_attn(query,normalized,normalized,mask)
        x=x+self.feed_forward(self.norm_ff(x))
        return torch.cat([cache,x],dim=1) if cache is not None else x


class EnglishTransformerLM(nn.Module,BatchScorerInterface):
    def __init__(self,config):
        super().__init__()
        if config['model_module']!='transformer' or config['pos_enc']!='none' or config.get('tie_weights'):
            raise ValueError('Unsupported language-model architecture')
        width=config['att_unit'];embedding=config['embed_unit'];vocab=config['n_vocab']
        self.embed=nn.Embedding(vocab,embedding)
        self.encoder=nn.Module()
        self.encoder.embed=nn.Sequential(nn.Linear(embedding,width),nn.LayerNorm(width),nn.Dropout(0.),nn.ReLU(),nn.Sequential())
        self.encoder.encoders=nn.ModuleList([LMLayer(width,config['head'],config['unit']) for _ in range(config['layer'])])
        self.encoder.after_norm=LayerNorm(width)
        self.decoder=nn.Linear(width,vocab)

    def logits(self,tokens,cache=None):
        mask=(tokens!=0).unsqueeze(1)&torch.ones(tokens.shape[1],tokens.shape[1],dtype=torch.bool,device=tokens.device).tril().unsqueeze(0)
        x=self.encoder.embed(self.embed(tokens));states=[]
        for i,layer in enumerate(self.encoder.encoders):
            x=layer(x,mask,None if cache is None else cache[i]);states.append(x)
        return self.decoder(self.encoder.after_norm(x)),states

    def batch_score(self,ys,states,xs):
        cache=None if states[0] is None else [torch.stack([s[i] for s in states]) for i in range(len(self.encoder.encoders))]
        logits,cache=self.logits(ys,cache)
        return logits[:,-1].log_softmax(-1),[[layer[b] for layer in cache] for b in range(len(ys))]

    def score(self,y,state,x):
        scores,states=self.batch_score(y.unsqueeze(0),[state],x.unsqueeze(0))
        return scores[0],states[0]


def load_language_model(directory,tokens):
    directory=Path(directory);config=json.loads((directory/'model.json').read_text())
    metadata=json.loads((directory/'provenance.json').read_text())
    vocab_hash=hashlib.sha256(json.dumps(tokens,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if config['n_vocab']!=len(tokens) or metadata['token_list_sha256']!=vocab_hash:
        raise ValueError('Language-model token vocabulary does not match recognition model')
    for filename in ['model.pth','model.json']:
        with (directory/filename).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=metadata['sha256'][filename]:raise ValueError('Language-model file checksum mismatch')
    lm=EnglishTransformerLM(config)
    lm.load_state_dict(torch.load(directory/'model.pth',map_location='cpu',weights_only=True),strict=True)
    return lm.eval()
