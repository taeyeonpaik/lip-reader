"""CPU correctness tests; no production training or paid compute."""
import copy
import random
import sys
import multiprocessing
import os
import signal
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/auto_avsr'))
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from training import batching, batched_model

torch.set_num_threads(2)


def graceful_worker_probe(ready, finish):
    batching.worker_init(0)
    ready.set()
    finish.wait(10)


def test_worker_sigterm_allows_graceful_parent_checkpoint():
    context = multiprocessing.get_context('spawn')
    ready, finish = context.Event(), context.Event()
    worker = context.Process(target=graceful_worker_probe, args=(ready, finish))
    worker.start()
    try:
        assert ready.wait(15)
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(.2)
        assert worker.is_alive()
        finish.set()
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()


class SmallFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 8, (5, 1, 1), padding=(2, 0, 0))
        self.bn = torch.nn.BatchNorm3d(8)

    def forward(self, x):
        return self.bn(self.conv(x.transpose(1, 2))).mean((-1, -2)).transpose(1, 2)


class SmallAuthorModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = SmallFrontend()
        self.proj_encoder = torch.nn.Linear(8, 8)
        self.encoder = ConformerEncoder(attention_dim=8, attention_heads=2, linear_units=16, num_blocks=2,
                                        dropout_rate=0, positional_dropout_rate=0, cnn_module_kernel=5)
        self.decoder = TransformerDecoder(11, attention_dim=8, attention_heads=2, linear_units=16, num_blocks=2,
                                          dropout_rate=0, positional_dropout_rate=0, self_attention_dropout_rate=0,
                                          src_attention_dropout_rate=0)
        self.ctc = CTC(11, 8, 0)
        self.criterion = LabelSmoothingLoss(11, -1, .1, False)
        self.sos = self.eos = 10
        self.ignore_id = -1


def samples(n=6):
    torch.manual_seed(91)
    return [dict(id=str(i), inputs=torch.randn(7+i, 1, 2, 2),
                 labels=torch.tensor([1, 2, 2] if i % 2 else [2, 3])) for i in range(n)]


def original_loss(model, clip):
    x = model.proj_encoder(model.frontend(clip['inputs'][None]))
    x, _ = model.encoder(x, None)
    labels = clip['labels'][None]
    ctc, _ = model.ctc(x, torch.tensor([x.shape[1]]), labels)
    inp, out = add_sos_eos(labels, model.sos, model.eos, -1)
    pred, _ = model.decoder(inp, target_mask(inp, -1), x, None)
    return .1*ctc+.9*model.criterion(pred, out)


def test_padded_outputs_match_original_and_ignore_padding():
    torch.manual_seed(3)
    model = SmallAuthorModules().eval()
    clips = samples(4)
    batch = batching.collate(clips)
    with torch.no_grad():
        encoded, _ = batched_model.encode(model, batch['inputs'], batch['lengths'])
        for i, clip in enumerate(clips):
            x = model.proj_encoder(model.frontend(clip['inputs'][None]))
            expected, _ = model.encoder(x, None)
            torch.testing.assert_close(encoded[i, :len(clip['inputs'])], expected[0], rtol=2e-5, atol=2e-6)
        for i, length in enumerate(batch['lengths']):
            batch['inputs'][i, length:] = 1000
        changed, _ = batched_model.encode(model, batch['inputs'], batch['lengths'])
        torch.testing.assert_close(encoded, changed)
    assert all(not layer.conv_module.depthwise_conv._forward_pre_hooks for layer in model.encoder.encoders)


@pytest.mark.parametrize('count', [6, 16])
def test_accumulated_loss_and_all_gradients_match_original(count):
    torch.manual_seed(2)
    model = SmallAuthorModules().eval()
    reference = copy.deepcopy(model)
    clips = samples(count)
    total = 0.
    for start in range(0, count, 4):
        subset = clips[start:start+4]
        value = batched_model.loss(model, batching.collate(subset), 'cpu')*len(subset)/count
        total += float(value.detach())
        value.backward()
    original_total = 0.
    for clip in clips:
        value = original_loss(reference, clip)/count
        original_total += float(value.detach())
        value.backward()
    assert total == pytest.approx(original_total, rel=2e-6)
    for (name, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=2e-6, msg=name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .98), weight_decay=.03)
    other_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-5, betas=(.9, .98), weight_decay=.03)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    optimizer.step()
    other_optimizer.step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=2e-6)


def test_bucket_coverage_tail_and_resume():
    rows = [dict(id=str(i), declared_frames=13+(i*31)%142) for i in range(45814)]
    batches = batching.epoch_batches(rows, 1)
    flattened = [i for batch in batches for i in batch]
    assert sorted(flattened) == list(range(len(rows)))
    assert all(len(b) == 4 for b in batches[:-1]) and len(batches[-1]) == 2
    assert batches == batching.epoch_batches(rows, 1)
    assert batches != batching.epoch_batches(rows, 2)
    assert batching.resume_batches(batches, 1600) == batches[400:]
    assert batching.resume_batches(batches, len(rows)) == []
    with pytest.raises(ValueError):
        batching.resume_batches(batches, 1)
    padded = sum(max(rows[i]['declared_frames'] for i in b)*len(b) for b in batches)
    rng = random.Random(124)
    rng.shuffle(rows)
    naive = sum(max(r['declared_frames'] for r in rows[j:j+4])*len(rows[j:j+4]) for j in range(0, len(rows), 4))
    assert padded < naive*.8


def test_augmentation_matches_original_distribution_and_semantics():
    seed = 201
    frames = np.random.default_rng(8).integers(0, 256, (51, 96, 96, 3), dtype=np.uint8)
    actual = batching.augment(frames, seed)
    torch.manual_seed(seed)
    random.seed(seed)
    top, left = torch.randint(0, 9, (2,)).tolist()
    rgb = torch.from_numpy(frames[:, top:top+88, left:left+88].copy()).float()/255.
    x = rgb[..., 0]*.2989+rgb[..., 1]*.587+rgb[..., 2]*.114
    for t, end in torch.randint(0, 10, (int((len(x)+25-.1)//25), 2)).tolist():
        if len(x)-t <= 0:
            continue
        start = random.randrange(0, len(x)-t)
        if t:
            x[start:start+end] = 0
    torch.testing.assert_close(actual, ((x-.421)/.165).unsqueeze(1), rtol=0, atol=0)


def test_workers_and_resume_preserve_preprocessing(tmp_path):
    rows = []
    for i in range(22):
        path = tmp_path/f'{i}.avi'
        frames = 8+i%5
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (96, 96))
        assert writer.isOpened()
        for _ in range(frames):
            writer.write(np.full((96, 96, 3), 30+i*5, dtype=np.uint8))
        writer.release()
        rows.append(dict(id=str(i), split='train', path=str(path), declared_frames=frames))
    tokens = [[1, 2] for _ in rows]
    single = list(batching.loader(rows, tokens, 3, workers=0, pin_memory=False))
    parallel = list(batching.loader(rows, tokens, 3, workers=2, pin_memory=False))
    resumed = list(batching.loader(rows, tokens, 3, offset=16, workers=2, pin_memory=False))
    for a, b in zip(single, parallel):
        assert a['ids'] == b['ids']
        torch.testing.assert_close(a['inputs'], b['inputs'], rtol=0, atol=0)
    for a, b in zip(single[4:], resumed):
        assert a['ids'] == b['ids']
        torch.testing.assert_close(a['inputs'], b['inputs'], rtol=0, atol=0)
    assert len(resumed) == 2


def test_frozen_bn_required():
    model = SmallAuthorModules().train()
    batch = batching.collate(samples(4))
    with pytest.raises(ValueError, match='frozen BN'):
        batched_model.encode(model, batch['inputs'], batch['lengths'])


def test_training_loop_atomic_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from training import lrs2_batched as trainer
    from training import lrs2_cuda as base
    clips = samples(22)
    rows = [dict(id=str(i), split='train', declared_frames=len(c['inputs'])) for i, c in enumerate(clips)]
    def fake_loader(rows, tokens, epoch, offset):
        return [batching.collate([clips[i] for i in indices])
                for indices in batching.resume_batches(batching.epoch_batches(rows, epoch), offset)]
    monkeypatch.setattr(batching, 'loader', fake_loader)
    monkeypatch.setattr(base, 'DEVICE', 'cpu')
    monkeypatch.setattr(base, 'RUN', tmp_path)
    monkeypatch.setattr(base, 'status', lambda *a, **k: None)
    monkeypatch.setattr(base, 'STOP_REQUESTED', False)
    torch.manual_seed(9)
    initial = SmallAuthorModules()
    for module in initial.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = .15
    class Harness(trainer.Experiment):
        def __init__(self):
            self.model = copy.deepcopy(initial)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, betas=(.9, .98), weight_decay=.03, foreach=False)
            self.splits = {'train': rows}
            self.tokens = [c['labels'].tolist() for c in clips]
        def mode(self, training):
            self.model.train(training)
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        def snapshot(self, state):
            torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), progress=state), tmp_path/'last.pt')
    state = dict(epoch=1, offset=0, updates=0, loss_sum=0.)
    full = Harness()
    full_state = copy.deepcopy(state)
    assert full.train_epoch(full_state, 1)
    interrupted = Harness()
    monkeypatch.setattr(base, 'STOP_REQUESTED', True)
    assert not interrupted.train_epoch(state, 1)
    assert state['offset'] == 16 and state['updates'] == 1
    saved = torch.load(tmp_path/'last.pt', weights_only=True)
    resumed = Harness()
    resumed.model.load_state_dict(saved['model'], strict=True)
    resumed.optimizer.load_state_dict(saved['optimizer'])
    monkeypatch.setattr(base, 'STOP_REQUESTED', False)
    assert resumed.train_epoch(saved['progress'], 1)
    assert saved['progress'] == full_state
    for name, tensor in full.model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
    before = copy.deepcopy(resumed.model.state_dict())
    assert resumed.train_epoch(saved['progress'], 1)
    assert saved['progress']['updates'] == 2
    for name, tensor in before.items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_configuration_supports_safe_reload(tmp_path, monkeypatch):
    from training import lrs2_cuda as base
    from training.lrs2_batched import Experiment
    def init_without_large_model(self, rows):
        self.config = {'runtime_torch': torch.__version__}
        self.splits = {'train': []}
        self.model = SmallAuthorModules().eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
    monkeypatch.setattr(base.Experiment, '__init__', init_without_large_model)
    monkeypatch.setattr(base, 'STORAGE', tmp_path)
    experiment = Experiment([])
    assert type(experiment.config['runtime_torch']) is str
    batched_model.loss(experiment.model, batching.collate(samples(4)), 'cpu').backward()
    experiment.optimizer.step()
    experiment.snapshot(dict(epoch=1, offset=16, updates=1))
    saved = torch.load(tmp_path/'last.pt', map_location='cpu', weights_only=True)
    assert saved['config'] == experiment.config
    experiment.model.load_state_dict(saved['model'], strict=True)
    experiment.optimizer.load_state_dict(saved['optimizer'])
