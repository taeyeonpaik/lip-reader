"""Three discarded CUDA updates. Run only inside the bounded shell wrapper."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'outputs/lipreading-studio'))
import torch
from torch.utils.data import DataLoader
from training import lrs2_cuda as base
from training import batching, batched_model
from training.lrs2_batched import Experiment


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no training started')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Read the previously verified cloud manifest; do not mutate its run state.
    rows = json.loads((base.RUN/'manifest.json').read_text())
    check = json.loads((base.RUN/'data-check.json').read_text())
    assert check['decoded_verified'] and check['sha256_verified']
    assert check['manifest_sha256'] == base.sha(base.RUN/'manifest.json')
    train = sorted([r for r in rows if r['split'] == 'train'], key=lambda r:(r['declared_frames'], r['id']))
    selected = [train[min(len(train)-1, int((i+.5)*len(train)/16))] for i in range(16)]
    # Verify every source used in this diagnostic, even though full data was verified earlier.
    for row in selected + [train[-1]]:
        base.verify_cache(row)
    experiment = Experiment(rows)
    experiment.mode(True)
    model = experiment.model
    all_tokens = {row['id']: ids for row, ids in zip(experiment.splits['train'], experiment.tokens)}
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if name in ('frontend.frontend3D.0.weight', 'proj_encoder.weight',
                          'encoder.encoders.0.feed_forward.w_1.weight', 'decoder.output_layer.weight', 'ctc.ctc_lo.weight')}
    # Longest four-clip padded batch probes memory without applying an update.
    probe_row = train[-1]
    probe_ds = batching.ClipDataset([probe_row]*4, [all_tokens[probe_row['id']]]*4, 1)
    probe = batching.collate([probe_ds[i] for i in range(4)])
    probe_loss = batched_model.loss(model, probe, 'cuda')
    probe_loss.backward()
    assert torch.isfinite(probe_loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    experiment.optimizer.zero_grad(set_to_none=True)
    del probe, probe_ds, probe_loss
    gc.collect()
    repeated = selected*3
    ds = batching.ClipDataset(repeated, [all_tokens[r['id']] for r in repeated], 1)
    loader = DataLoader(ds, batch_sampler=[list(range(i, i+4)) for i in range(0, 48, 4)],
                        num_workers=2, prefetch_factor=2, pin_memory=True, timeout=60,
                        multiprocessing_context='spawn', worker_init_fn=batching.worker_init,
                        collate_fn=batching.collate, generator=torch.Generator().manual_seed(124))
    iterator = iter(loader)
    timings = []
    for update in range(3):
        base.seed(123+update)
        torch.cuda.synchronize()
        start = time.monotonic()
        experiment.optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            batch = next(iterator)
            loss = batched_model.loss(model, batch, 'cuda')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark loss')
            (loss/4).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        experiment.optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
        print(json.dumps(dict(update=update+1, seconds=timings[-1], gradient_norm=float(norm))), flush=True)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    changed = {name: not torch.equal(old, dict(model.named_parameters())[name]) for name, old in before.items()}
    assert all(changed.values())
    assert base.sha(base.ROOT/'models/autoavsr.pth') == base.BASE_SHA256
    seconds_per_clip = statistics.median(timings[1:])/16
    report = dict(passed=True, device=torch.cuda.get_device_name(), torch=torch.__version__, microbatch=4,
                  effective_batch=16, precision='FP32; TF32 disabled', workers=2,
                  batch_seconds=timings, seconds_per_clip=seconds_per_clip,
                  projected_training_hours=seconds_per_clip*len(train)/3600,
                  frames=[r['declared_frames'] for r in selected], longest_batch_frames=probe_row['declared_frames'],
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                  representative_weights_changed=changed, all_gradients_finite=True,
                  updates_discarded=True, original_weights_unchanged=True,
                  validation_or_test_used=False, production_checkpoint_written=False,
                  limitations='Short repeated training sample; excludes validation/checkpoint time. Not a sustained epoch measurement.')
    base.atomic_json(ROOT/'work/lightning/batched-test-result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
