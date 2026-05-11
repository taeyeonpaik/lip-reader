"""Standard full-model Auto-AVSR fine-tuning on verified LRS2 splits only.

Unchanged author architecture and hybrid CTC/attention objective. The visual
network runs on MPS and the text decoder/CTC run on CPU, preserving gradients.
Large checkpoints live outside Documents. Test evaluation follows selection.
"""
import argparse
import fcntl
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import sentencepiece as spm
import torch

from studio.pipeline import ROOT, VENDOR, LipReader
from studio.adaptation import BASE_SHA256
from training.data import atomic_json, validate_splits
from training.inputs import load_pixels
from training.metrics import summarize, normalize
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask

WORKSPACE = ROOT.parents[1]
RUN = ROOT / 'training/runs/lrs2-full-v1'
STORAGE = Path('/private/tmp/autoavsr-lrs2-full-v1')
SOURCE_MANIFEST = WORKSPACE / 'work/corpora/lrs2/manifest.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def vendor_sha():
    digest=hashlib.sha256()
    for path in sorted((VENDOR/'espnet').rglob('*.py')):
        digest.update(str(path.relative_to(VENDOR)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def read_rgb(row):
    cap=cv2.VideoCapture(row['path']); frames=[]
    try:
        assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01, 'Wrong frame rate: '+row['id']
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape==(96,96,3), 'Invalid mouth crop: '+row['id']
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    assert len(frames)==row['declared_frames'] and frames, 'Wrong frame count: '+row['id']
    return np.stack(frames)


def verify_cache(row):
    assert sha(row['path'])==row['sha256'], 'Source video changed: '+row['id']
    frames=read_rgb(row)
    return hashlib.sha256(frames.tobytes()).hexdigest()


def verify_source(row):
    assert sha(row['path'])==row['sha256'], 'Source video changed: '+row['id']
    return row['sha256']


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(value)


def save_torch(value, path):
    # One complete, atomic checkpoint contains weights, optimizer and progress.
    path = Path(path)
    if shutil.disk_usage(path.parent).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; refusing checkpoint write')
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    with tmp.open('rb') as f: os.fsync(f.fileno())
    tmp.replace(path)


def status(stage, **values):
    report = dict(state='running', stage=stage, updated_unix=time.time(), **values)
    atomic_json(RUN / 'progress.json', report)
    print(stage, json.dumps(values), flush=True)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True); STORAGE.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE_MANIFEST.read_text())
    rows=[dict(r,dataset='LRS2',format='rgb-mp4',session=r['id'].split('/')[1],
               evaluation_group=r['id'].split('/')[1]) for r in source]
    validate_splits(rows)
    assert {s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}==dict(train=45814,validation=1082,test=1243)
    sessions={s:{r['session'] for r in rows if r['split']==s} for s in ('train','validation','test')}
    assert not sessions['train'] & sessions['validation']
    assert not sessions['train'] & sessions['test']
    assert not sessions['validation'] & sessions['test']
    assert all(Path(r['path']).is_file() for r in rows)
    path=RUN/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==rows, 'Manifest changed'
    else: atomic_json(path,rows)
    report=dict(source_manifest_sha256=sha(SOURCE_MANIFEST),
                source_provenance_sha256=sha(SOURCE_MANIFEST.with_name('provenance.json')),
                splits={s:sum(r['split']==s for r in rows) for s in sessions},
                recording_overlap=0,clip_overlap=0,original_distributor_splits_preserved=True,
                official_counts=dict(train=45839,validation=1082,test=1243),
                official_url='https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html',
                limitations=['Official raw files and lists require BBC credentials; they were not downloaded or bypassed.',
                            'Uses existing public processed crop distribution. Training contains 25 fewer clips than the official count; reason not independently established.',
                            'Validation and test counts match official counts; independent official filelist identity comparison unavailable.',
                            'Pretrained checkpoint already trained on LRS2 training data; this alone does not prove exposure to official validation/test.',
                            'Prior LRS2 test results are already known. No test-based selection in this run.'])
    if (RUN/'dataset-protocol.json').exists(): assert json.loads((RUN/'dataset-protocol.json').read_text())==report
    else: atomic_json(RUN/'dataset-protocol.json',report)
    return rows


def pixels(row, training=False):
    if not training: return load_pixels(row, 'mps')
    frames = read_rgb(row)
    # Author RandomCrop(88) uses one spatial offset for a whole utterance.
    top, left = torch.randint(0, 9, (2,)).tolist()
    rgb = torch.from_numpy(frames[:, top:top+88, left:left+88].copy()).float() / 255.
    x = rgb[...,0]*.2989 + rgb[...,1]*.587 + rgb[...,2]*.114
    # Exact author AdaptiveTimeMask(10, 25) semantics, before normalization.
    n_mask = int((len(x) + 25 - .1) // 25)
    for t, end in torch.randint(0, 10, size=(n_mask, 2)).tolist():
        if len(x) - t <= 0: continue
        start = random.randrange(0, len(x)-t)
        if t: x[start:start+end] = 0
    return ((x-.421)/.165).unsqueeze(0).unsqueeze(2).to('mps')


class Experiment:
    def __init__(self, rows):
        self.rows = rows
        self.splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
        assert sha(ROOT/'models/autoavsr.pth') == BASE_SHA256
        seed(123)
        self.reader = LipReader(device='mps', beam_size=40)
        self.model = self.reader.model
        for parameter in self.model.parameters(): parameter.requires_grad_(True)
        self.sp = spm.SentencePieceProcessor(model_file=str(VENDOR/'spm/unigram/unigram5000.model'))
        self.units = {line.split()[0]: int(line.split()[-1]) for line in (VENDOR/'spm/unigram/unigram5000_units.txt').read_text().splitlines()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, betas=(.9, .98), weight_decay=.03, foreach=False)
        self.config = dict(
            base_sha256=BASE_SHA256, manifest_sha256=sha(RUN/'manifest.json'),
            architecture='unchanged original Auto-AVSR E2E video', trainable='all model parameters',
            trainable_parameters=sum(p.numel() for p in self.model.parameters()),
            dataset='LRS2 only; all available main training crops and original distributor validation/test partitions',
            splits={s: len(v) for s,v in self.splits.items()}, seed=123,
            lr=1e-5, optimizer='AdamW', betas=[.9,.98], weight_decay=.03,
            microbatch=1, effective_batch=16, gradient_clip=1., max_epochs=20,
            minimum_epochs=3, patience=4, schedule='one epoch linear warmup then cosine decay',
            loss='author 0.1 CTC + 0.9 attention; original 0.1 label smoothing',
            batchnorm='frozen running statistics for microbatch 1; affine parameters trainable',
            augmentation='author random spatial crop 88 and AdaptiveTimeMask(10,25)',
            validation_transform='center88, /255, normalize(.421,.165)',
            decoder={'beam_size':40, 'ctc_weight':.1, 'lm_weight':0., 'length_bonus':0.},
            selection='lowest validation WER; earlier checkpoint wins ties; include original baseline',
            test_policy='no test gradients or model selection; evaluate baseline and selected model after selection',
            prior_test_exposure='beam5 pretrained test result is already known; author-default beam40 fixed before this training, no test sweep',
            pretraining_history='original checkpoint was pretrained on mixed corpora; this new run uses LRS2 only and starts from the same original author weights, not LRS3-adapted weights',
            checkpoints=str(STORAGE), source_sha256=sha(__file__),
            inputs_sha256=sha(ROOT/'training/inputs.py'), pipeline_sha256=sha(ROOT/'studio/pipeline.py'),
            tokenizer_sha256=sha(VENDOR/'spm/unigram/unigram5000.model'), vendor_sha256=vendor_sha())
        self.config.update(execution_training_cpu_threads=4,execution_training_decoder_device='cpu',execution_evaluation_cpu_threads=4,
                           decoded_verification_proof_sha256=sha(RUN/'decoded-verification-proof.json'))
        self.config.update(validation_origin='Original LRS2 validation partition from pinned processed distribution',
                           source_manifest_sha256=sha(SOURCE_MANIFEST),
                           dataset_protocol_sha256=sha(RUN/'dataset-protocol.json'),
                           changed_from_lrs3='Dataset and data-format handling only; no LRS3 weights or optimizer transferred')

    def mode(self, training):
        torch.set_num_threads(4 if training else 4)
        self.model.decoder.to('cpu' if training else 'cpu')
        self.model.criterion.to('cpu' if training else 'cpu')
        self.model.train(training)
        if training:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()

    def encode(self, x):
        x = self.model.proj_encoder(self.model.frontend(x))
        x, _ = self.model.encoder(x, None)
        return x.cpu()  # Autograd crosses the device boundary.

    def loss(self, row, augment=True):
        x = self.encode(pixels(row, augment))
        ids = [self.units.get(p, self.units['<unk>']) for p in self.sp.encode(row['text'], out_type=str)]
        if len(ids)+sum(a == b for a,b in zip(ids,ids[1:])) > x.shape[1]:
            raise ValueError('Reference too long for CTC: '+row['id'])
        labels = torch.tensor([ids], dtype=torch.long)
        length = torch.tensor([x.shape[1]])
        ctc, _ = self.model.ctc(x, length, labels)
        inp, out = add_sos_eos(labels, self.model.sos, self.model.eos, self.model.ignore_id)
        pred, _ = self.model.decoder(inp, target_mask(inp,self.model.ignore_id), x, None)
        return .1*ctc + .9*self.model.criterion(pred,out)

    def predict(self, row):
        with torch.inference_mode(): encoded = self.encode(pixels(row))
        return self.reader.decode(encoded.squeeze(0))

    def evaluate(self, split, name):
        self.mode(False)
        path = RUN/(name+'.jsonl'); records=[]
        if path.exists():
            lines = path.read_text().splitlines()
            for i,line in enumerate(lines):
                try: record=json.loads(line)
                except json.JSONDecodeError:
                    assert i == len(lines)-1; break
                row=self.splits[split][i]
                assert record['id']==row['id'] and record['reference']==row['text']
                records.append(record)
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with path.open('a') as f:
            for row in self.splits[split][len(records):]:
                tick=time.monotonic()
                record=dict(id=row['id'],session=row.get('session'),evaluation_group=row.get('evaluation_group'),reference=row['text'],prediction=self.predict(row))
                record['seconds']=time.monotonic()-tick
                records.append(record); f.write(json.dumps(record)+'\n'); f.flush()
                if len(records)%50==0 or len(records)==len(self.splits[split]):
                    status(name,completed=len(records),total=len(self.splits[split]))
        result=summarize(records)
        atomic_json(RUN/(name+'.json'),result)
        return result

    def snapshot(self, state):
        save_torch(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
                        progress=state,config=self.config),STORAGE/'last.pt')

    def execute(self):
        config_path=RUN/'config.json'
        if config_path.exists(): assert json.loads(config_path.read_text())==self.config, 'Config changed on resume'
        else: atomic_json(config_path,self.config)
        atomic_json(RUN/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,
                    sentencepiece=spm.__version__,device=self.reader.device,started_unix=time.time(),pid=os.getpid()))
        # Validate every cache and reference before starting gradients.
        hashes={}
        # File checks have no RNG or model access. Parallel I/O preserves the
        # manifest order and all source, shape, pixel and split checks.
        proof=json.loads((RUN/'decoded-verification-proof.json').read_text())
        assert proof['manifest_sha256']==sha(RUN/'manifest.json')
        assert proof['inputs_sha256']==sha(ROOT/'training/inputs.py')
        assert proof['original_data_check_sha256']==sha(RUN/'data-check-before-speedup.json')
        assert proof['original_config_sha256']==sha(RUN/'config-before-speedup.json')
        assert proof['original_training_source_sha256']==sha(RUN/'training-source-before-speedup.py')
        assert proof['original_data_check']['sha256_verified'] and proof['original_data_check']['clips']==len(self.rows)
        status('verify-source-hashes',completed=0,total=len(self.rows),workers=8)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,(row,digest) in enumerate(zip(self.rows,pool.map(verify_source,self.rows)),1):
                assert hashes.get(digest,row['split'])==row['split'], 'Duplicate pixels across splits'
                hashes[digest]=row['split']
                if i%1000==0: status('verify-source-hashes',completed=i,total=len(self.rows),workers=8)
        atomic_json(RUN/'data-check.json',dict(clips=len(self.rows),sha256_verified=True,lrs2_clips=len(self.rows),lrs3_clips=0,recording_overlap=0,decoded_checks_reused_from_verified_identical_source_bytes=True,proof_sha256=sha(RUN/'decoded-verification-proof.json')))
        baseline=self.evaluate('validation','baseline-validation')
        state=dict(epoch=1,offset=0,updates=0,loss_sum=0.,history=[],best_wer=baseline['wer'],best_epoch=0)
        if (STORAGE/'last.pt').exists():
            saved=torch.load(STORAGE/'last.pt',map_location='cpu',weights_only=True)
            assert saved['config']==self.config
            self.model.load_state_dict(saved['model'],strict=True)
            self.optimizer.load_state_dict(saved['optimizer']); state=saved['progress']
            del saved; gc.collect()
        batches=math.ceil(len(self.splits['train'])/16)
        for epoch in range(state['epoch'],21):
            if epoch>3 and epoch-1-state['best_epoch']>=4: break
            order=list(self.splits['train']); random.Random(123+epoch).shuffle(order)
            self.mode(True); tick=time.monotonic(); initial_offset=state['offset']
            for offset in range(state['offset'],len(order),16):
                # Deterministic per-update RNG lets a saved optimizer boundary resume exactly.
                seed(123+state['updates'])
                step=state['updates']; progress=step/batches
                scale=(step+1)/batches if progress<1 else .5*(1+math.cos(math.pi*(progress-1)/19))
                for group in self.optimizer.param_groups: group['lr']=1e-5*scale
                self.optimizer.zero_grad(set_to_none=True)
                batch=order[offset:offset+16]
                for row in batch:
                    loss=self.loss(row)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    state['loss_sum']+=float(loss.detach())
                    (loss/len(batch)).backward()
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
                self.optimizer.step()
                state.update(epoch=epoch,offset=offset+len(batch),updates=step+1)
                if state['updates']%10==0:
                    status('training',epoch=epoch,max_epochs=20,completed=state['offset'],total=len(order),
                           updates=state['updates'],mean_loss=state['loss_sum']/state['offset'],
                           gradient_norm=float(norm),lr=self.optimizer.param_groups[0]['lr'],
                           seconds_per_clip=(time.monotonic()-tick)/(state['offset']-initial_offset))
                if state['updates']%100==0: self.snapshot(state)
            self.optimizer.zero_grad(set_to_none=True)
            # Persist end-of-epoch weights before validation. Interrupted validation
            # can then resume against precisely those weights without retraining.
            self.snapshot(state)
            val=self.evaluate('validation',f'epoch-{epoch:02}-validation')
            state['history'].append(dict(epoch=epoch,loss=state['loss_sum']/len(order),validation=val,updates=state['updates']))
            if val['wer']<state['best_wer']:
                save_torch(self.model.state_dict(),STORAGE/'best.pth')
                state.update(best_wer=val['wer'],best_epoch=epoch)
            state.update(epoch=epoch+1,offset=0,loss_sum=0.)
            self.snapshot(state); atomic_json(RUN/'history.json',state['history'])
        selected=STORAGE/'best.pth' if state['best_epoch'] else ROOT/'models/autoavsr.pth'
        selection=dict(checkpoint=str(selected),sha256=sha(selected),epoch=state['best_epoch'],
                       baseline_validation=baseline,best_validation_wer=state['best_wer'],test_used_for_selection=False)
        selection_path=RUN/'selection.json'
        if selection_path.exists(): assert json.loads(selection_path.read_text())==selection
        else: atomic_json(selection_path,selection)
        # Selection is now final. Release optimizer state before final evaluation.
        del self.optimizer; gc.collect(); torch.mps.empty_cache()
        self.model.load_state_dict(torch.load(ROOT/'models/autoavsr.pth',map_location='cpu',weights_only=True),strict=True)
        baseline_test=self.evaluate('test','baseline-test')
        self.model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=True),strict=True)
        final_validation=self.evaluate('validation','selected-validation')
        final_test=self.evaluate('test','selected-test')
        assert final_validation['wer']==state['best_wer']
        assert sha(ROOT/'models/autoavsr.pth')==BASE_SHA256
        atomic_json(RUN/'result.json',dict(state='training_and_evaluation_complete_review_pending',selection=selection,
                    baseline_test=baseline_test,selected_test=final_test,selected_validation=final_validation,
                    epochs_completed=len(state['history']),original_weights_unchanged=True,
                    note='Independent metric, reload, uncertainty and product review still required. No app activation yet.'))
        status('review-ready',baseline_test_wer=baseline_test['wer'],selected_test_wer=final_test['wer'],selected_epoch=state['best_epoch'])


def smoke(rows):
    experiment=Experiment(rows)
    train=sorted(experiment.splits['train'],key=lambda r:(r['declared_frames'],r['id']))
    samples=[train[len(train)//2],train[-1]]
    experiment.mode(True)
    measurements=[]
    before={name:p.detach().clone() for name,p in experiment.model.named_parameters()
            if name in ('frontend.frontend3D.0.weight','proj_encoder.weight','encoder.encoders.0.feed_forward.w_1.weight','decoder.output_layer.weight','ctc.ctc_lo.weight')}
    for row in samples:
        tick=time.monotonic();experiment.optimizer.zero_grad(set_to_none=True)
        loss=experiment.loss(row);loss.backward()
        assert torch.isfinite(loss)
        groups={}
        for name,p in experiment.model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            group=name.split('.')[0];groups[group]=groups.get(group,0.)+float(p.grad.detach().float().norm().cpu())
        assert all(v>0 for v in groups.values())
        norm=torch.nn.utils.clip_grad_norm_(experiment.model.parameters(),1.,error_if_nonfinite=True)
        experiment.optimizer.step();torch.mps.synchronize()
        result=dict(id=row['id'],frames=row['declared_frames'],loss=float(loss.detach()),
                    gradient_norm=float(norm),seconds=time.monotonic()-tick,gradient_norm_by_component=groups,
                    mps_driver_gib=torch.mps.driver_allocated_memory()/1024**3)
        print(json.dumps(result),flush=True);measurements.append(result)
    changed={n:bool(not torch.equal(t,dict(experiment.model.named_parameters())[n])) for n,t in before.items()}
    assert changed and all(changed.values())
    report=dict(state='passed',trainable_parameters=experiment.config['trainable_parameters'],measurements=measurements,
                representative_weights_changed=changed,original_unchanged=sha(ROOT/'models/autoavsr.pth')==BASE_SHA256,
                smoke_updates_discarded=True)
    atomic_json(RUN/'feasibility.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('initializing')
        rows=prepare()
        if args.smoke: smoke(rows)
        elif (RUN/'result.json').exists(): print('Already complete; review reports.',flush=True)
        else: Experiment(rows).execute()
    except Exception as error:
        atomic_json(RUN/'progress.json',dict(state='failed',error=f'{type(error).__name__}: {error}',updated_unix=time.time()))
        raise


if __name__=='__main__': main()
