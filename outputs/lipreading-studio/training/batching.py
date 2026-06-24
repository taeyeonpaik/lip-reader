"""Deterministic CPU preprocessing and length-bucketed training batches."""
import hashlib
import random
import signal

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def epoch_batches(rows, epoch, microbatch=4, pool_size=256, seed=123):
    if pool_size % microbatch:
        raise ValueError('Bucket pool must be divisible by microbatch size')
    rng = random.Random(seed + epoch)
    order = list(range(len(rows)))
    rng.shuffle(order)
    full, tail = [], []
    for start in range(0, len(order), pool_size):
        pool = sorted(order[start:start+pool_size],
                      key=lambda i: (rows[i]['declared_frames'], rows[i]['id']))
        for offset in range(0, len(pool), microbatch):
            batch = pool[offset:offset+microbatch]
            (full if len(batch) == microbatch else tail).append(batch)
    rng.shuffle(full)
    return full + tail


def resume_batches(batches, offset):
    consumed = 0
    for index, batch in enumerate(batches):
        if consumed == offset:
            return batches[index:]
        consumed += len(batch)
    if consumed == offset:
        return []
    raise ValueError('Resume offset is not a batch boundary')


def augment(frames, seed):
    """Author crop/time-mask distribution with a private per-clip RNG."""
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    top, left = torch.randint(0, 9, (2,), generator=generator).tolist()
    rgb = torch.from_numpy(frames[:, top:top+88, left:left+88].copy()).float()/255.
    x = rgb[..., 0]*.2989 + rgb[..., 1]*.587 + rgb[..., 2]*.114
    n_mask = int((len(x) + 25 - .1)//25)
    for t, end in torch.randint(0, 10, (n_mask, 2), generator=generator).tolist():
        if len(x)-t <= 0:
            continue
        start = rng.randrange(0, len(x)-t)
        if t:
            x[start:start+end] = 0
    return ((x-.421)/.165).unsqueeze(1)


class ClipDataset(Dataset):
    def __init__(self, rows, tokens, epoch, seed=123):
        if len(rows) != len(tokens) or any(r['split'] != 'train' for r in rows):
            raise ValueError('Training loader accepts training rows only')
        self.rows, self.tokens, self.epoch, self.seed = rows, tokens, epoch, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cap = cv2.VideoCapture(row['path'])
        frames = []
        try:
            if abs(cap.get(cv2.CAP_PROP_FPS)-25) >= .01:
                raise ValueError('Wrong FPS: '+row['id'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (96, 96, 3):
                    raise ValueError('Invalid crop: '+row['id'])
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames or len(frames) != row['declared_frames']:
            raise ValueError('Wrong frame count: '+row['id'])
        digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{row["id"]}'.encode()).digest()
        sample_seed = int.from_bytes(digest[:8], 'big') % (2**63)
        return dict(id=row['id'], inputs=augment(np.stack(frames), sample_seed),
                    labels=torch.tensor(self.tokens[index], dtype=torch.long))


def collate(clips):
    lengths = torch.tensor([len(c['inputs']) for c in clips], dtype=torch.long)
    label_lengths = torch.tensor([len(c['labels']) for c in clips], dtype=torch.long)
    for clip, frames in zip(clips, lengths.tolist()):
        tokens = clip['labels'].tolist()
        if not tokens or len(tokens)+sum(a == b for a, b in zip(tokens, tokens[1:])) > frames:
            raise ValueError('Invalid CTC reference: '+clip['id'])
    return dict(ids=[c['id'] for c in clips], lengths=lengths, label_lengths=label_lengths,
                inputs=torch.nn.utils.rnn.pad_sequence([c['inputs'] for c in clips], batch_first=True),
                labels=torch.nn.utils.rnn.pad_sequence([c['labels'] for c in clips], batch_first=True, padding_value=-1))




def worker_init(_):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def loader(rows, tokens, epoch, offset=0, workers=2, pin_memory=True):
    if offset != len(rows) and offset % 16:
        raise ValueError('Resume requires a completed optimizer update')
    batches = resume_batches(epoch_batches(rows, epoch), offset)
    options = dict(num_workers=workers, pin_memory=pin_memory, collate_fn=collate,
                   generator=torch.Generator().manual_seed(123+epoch))
    if workers:
        options.update(prefetch_factor=2, multiprocessing_context='spawn', worker_init_fn=worker_init, timeout=60)
    return DataLoader(ClipDataset(rows, tokens, epoch), batch_sampler=batches, **options)
