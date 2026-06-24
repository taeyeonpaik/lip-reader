"""Padding-safe execution of the unchanged author modules and hybrid loss.

Requires frozen batch-normalization statistics. Temporary hooks zero padding
after the Conformer pointwise GLU and before each temporal convolution, so
padding cannot flow back into real frames. No parameters or vendor files change.
"""
from contextlib import contextmanager

import torch


@contextmanager
def temporal_masks(encoder, valid):
    handles = []
    def mask_input(module, args):
        return (args[0].masked_fill(~valid[:, None, :], 0),)
    try:
        for layer in encoder.encoders:
            if layer.conv_module is not None:
                handles.append(layer.conv_module.depthwise_conv.register_forward_pre_hook(mask_input))
        yield
    finally:
        for handle in handles:
            handle.remove()


def encode(model, inputs, lengths):
    if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
        raise ValueError('Batched variable-length training requires frozen BN statistics')
    valid = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths[:, None]
    inputs = inputs.masked_fill(~valid[:, :, None, None, None], 0)
    features = model.proj_encoder(model.frontend(inputs))
    if features.shape[:2] != valid.shape:
        raise ValueError('Unexpected temporal downsampling')
    with temporal_masks(model.encoder, valid):
        features, _ = model.encoder(features, valid[:, None, :])
    return features, valid[:, None, :]


def loss(model, batch, device):
    # Imported after the application's vendor path initialization.
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask
    tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()
               if torch.is_tensor(value)}
    encoded, memory_mask = encode(model, tensors['inputs'], tensors['lengths'])
    labels = tensors['labels']
    ctc, _ = model.ctc(encoded, tensors['lengths'], labels)
    inp, out = add_sos_eos(labels, model.sos, model.eos, model.ignore_id)
    valid_target = torch.arange(inp.shape[1], device=device)[None, :] < (tensors['label_lengths']+1)[:, None]
    mask = target_mask(inp, model.ignore_id) & valid_target[:, None, :]
    pred, _ = model.decoder(inp, mask, encoded, memory_mask)
    # Both author losses are sums over each utterance, averaged over the batch.
    if model.criterion.normalize_length:
        raise ValueError('Expected original per-utterance label smoothing reduction')
    return .1*ctc + .9*model.criterion(pred, out)
