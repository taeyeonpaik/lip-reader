"""Four-clip CUDA training with deterministic length buckets and CPU prefetch.

Separate run/checkpoints preserve the original one-clip trainer as a fallback.
Run with python -m training.lrs2_batched; --verify-only requires no GPU.
"""
import argparse
import fcntl
import gc
import itertools
import json
import math
import os
import signal
import time

import torch

from training import lrs2_cuda as base
from training import batching, batched_model
from training.data import atomic_json


def configure():
    base.RUN = base.ROOT/'training/runs/lrs2-batched-v1'
    base.STORAGE = base.WORKSPACE/'checkpoints/lrs2-batched-v1'


class Experiment(base.Experiment):
    def __init__(self, rows):
        super().__init__(rows)
        self.tokens = [[self.units.get(p, self.units['<unk>']) for p in self.sp.encode(row['text'], out_type=str)]
                       for row in self.splits['train']]
        self.config.update(microbatch=4, accumulation_steps=4, effective_batch=16,
                           runtime_torch=str(torch.__version__),
                           batchnorm='frozen running statistics; affine parameters trainable',
                           data_workers=2, prefetch_factor=2, pin_memory=True,
                           order='seeded shuffle; length-sort pools of 256; shuffle full four-clip batches; partial batch last',
                           augmentation_rng='SHA256(seed:epoch:clip_id); independent of workers and prefetch',
                           padding='zero normalized input; encoder/decoder attention masks; pre-depthwise Conformer mask; true CTC lengths',
                           accumulation='per-utterance weighting including partial final update',
                           batched_source_sha256=base.sha(__file__),
                           batching_sha256=base.sha(batching.__file__),
                           batched_model_sha256=base.sha(batched_model.__file__))
        self.config.update(execution_evaluation_decoder_device='cuda', execution_evaluation_cpu_threads=1)

    def mode(self, training):
        super().mode(training)
        if not training:
            # Beam hypotheses already batch together in the author's search.
            # Keep decoder/CTC on CUDA instead of the inherited Mac CPU path.
            torch.set_num_threads(1)
            self.model.decoder.to(base.DEVICE)
            self.model.ctc.to(base.DEVICE)
            self.reader.search.to(base.DEVICE)

    def predict(self, row):
        with torch.inference_mode():
            encoded = self.encode(base.pixels(row)).squeeze(0)
            hypotheses = self.reader.search(encoded)
            if not hypotheses:
                return ''
            ids = hypotheses[0].yseq.tolist()[1:]
            return ''.join(self.reader.tokens[i] for i in ids
                           if 0 < i < len(self.reader.tokens)-1).replace('▁', ' ').strip()

    def train_epoch(self, state, epoch):
        rows = self.splits['train']
        data = iter(batching.loader(rows, self.tokens, epoch, state['offset']))
        self.mode(True)
        tick, initial_offset = time.monotonic(), state['offset']
        updates_per_epoch = math.ceil(len(rows)/16)
        while state['offset'] < len(rows):
            group = list(itertools.islice(data, 4))
            count = sum(len(batch['ids']) for batch in group)
            if count != min(16, len(rows)-state['offset']):
                raise RuntimeError('Unexpected effective batch size')
            base.seed(123+state['updates'])
            step = state['updates']
            progress = step/updates_per_epoch
            scale = (step+1)/updates_per_epoch if progress < 1 else .5*(1+math.cos(math.pi*(progress-1)/19))
            for optimizer_group in self.optimizer.param_groups:
                optimizer_group['lr'] = 1e-5*scale
            self.optimizer.zero_grad(set_to_none=True)
            for batch in group:
                loss = batched_model.loss(self.model, batch, base.DEVICE)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                micro_count = len(batch['ids'])
                state['loss_sum'] += float(loss.detach())*micro_count
                (loss*(micro_count/count)).backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            state.update(epoch=epoch, offset=state['offset']+count, updates=step+1)
            if state['updates'] % 10 == 0 or state['offset'] == len(rows):
                base.status('training', epoch=epoch, max_epochs=20, completed=state['offset'], total=len(rows),
                            updates=state['updates'], mean_loss=state['loss_sum']/state['offset'],
                            gradient_norm=float(norm), lr=self.optimizer.param_groups[0]['lr'],
                            seconds_per_clip=(time.monotonic()-tick)/(state['offset']-initial_offset),
                            microbatch=4, effective_batch=16)
            if state['updates'] % 100 == 0 or base.STOP_REQUESTED:
                self.snapshot(state)
            if base.STOP_REQUESTED:
                atomic_json(base.RUN/'progress.json', dict(state='paused', stage='training', updated_unix=time.time(),
                            epoch=epoch, completed=state['offset'], total=len(rows), updates=state['updates'],
                            reason='Atomic checkpoint saved at completed optimizer update.'))
                return False
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def execute(self):
        config_path = base.RUN/'config.json'
        if config_path.exists():
            assert json.loads(config_path.read_text()) == self.config, 'Config changed on resume'
        else:
            atomic_json(config_path, self.config)
        atomic_json(base.RUN/'environment.json', dict(torch=torch.__version__, device=self.reader.device,
                    started_unix=time.time(), pid=os.getpid()))
        base.verify_all(self.rows)
        baseline = self.evaluate('validation', 'baseline-validation')
        state = dict(epoch=1, offset=0, updates=0, loss_sum=0., history=[], best_wer=baseline['wer'], best_epoch=0)
        if (base.STORAGE/'last.pt').exists():
            saved = torch.load(base.STORAGE/'last.pt', map_location='cpu', weights_only=True)
            assert saved['config'] == self.config
            self.model.load_state_dict(saved['model'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            state = saved['progress']
            del saved
            gc.collect()
        for epoch in range(state['epoch'], 21):
            if epoch > 3 and epoch-1-state['best_epoch'] >= 4:
                break
            if not self.train_epoch(state, epoch):
                return
            self.snapshot(state)
            val = self.evaluate('validation', f'epoch-{epoch:02}-validation')
            state['history'].append(dict(epoch=epoch, loss=state['loss_sum']/len(self.splits['train']),
                                         validation=val, updates=state['updates']))
            if val['wer'] < state['best_wer']:
                base.save_torch(self.model.state_dict(), base.STORAGE/'best.pth')
                state.update(best_wer=val['wer'], best_epoch=epoch)
            state.update(epoch=epoch+1, offset=0, loss_sum=0.)
            self.snapshot(state)
            atomic_json(base.RUN/'history.json', state['history'])
        selected = base.STORAGE/'best.pth' if state['best_epoch'] else base.ROOT/'models/autoavsr.pth'
        selection = dict(checkpoint=str(selected), sha256=base.sha(selected), epoch=state['best_epoch'],
                         baseline_validation=baseline, best_validation_wer=state['best_wer'], test_used_for_selection=False)
        selection_path = base.RUN/'selection.json'
        if selection_path.exists():
            assert json.loads(selection_path.read_text()) == selection
        else:
            atomic_json(selection_path, selection)
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(base.ROOT/'models/autoavsr.pth', map_location='cpu', weights_only=True), strict=True)
        baseline_test = self.evaluate('test', 'baseline-test')
        self.model.load_state_dict(torch.load(selected, map_location='cpu', weights_only=True), strict=True)
        final_validation = self.evaluate('validation', 'selected-validation')
        final_test = self.evaluate('test', 'selected-test')
        assert final_validation['wer'] == state['best_wer']
        assert base.sha(base.ROOT/'models/autoavsr.pth') == base.BASE_SHA256
        atomic_json(base.RUN/'result.json', dict(state='training_and_evaluation_complete_review_pending', selection=selection,
                    baseline_test=baseline_test, selected_test=final_test, selected_validation=final_validation,
                    epochs_completed=len(state['history']), original_weights_unchanged=True,
                    note='Independent metric, reload, uncertainty and product review still required. No app activation yet.'))
        base.status('review-ready', baseline_test_wer=baseline_test['wer'], selected_test_wer=final_test['wer'], selected_epoch=state['best_epoch'])


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, base.request_stop)
    signal.signal(signal.SIGINT, base.request_stop)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.verify_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; training was not started.')
    base.RUN.mkdir(parents=True, exist_ok=True)
    with (base.RUN/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows = base.prepare()
            if args.verify_only:
                base.verify_all(rows)
            elif (base.RUN/'result.json').exists():
                print('Already complete; review reports.', flush=True)
            else:
                Experiment(rows).execute()
        except Exception as error:
            atomic_json(base.RUN/'progress.json', dict(state='failed', error=f'{type(error).__name__}: {error}', updated_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
