"""Training, evaluation, self-contained checkpoints, and synthetic validation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PreprocessConfig, VideoDataset, collate_batch
from .model import LipNet, ModelConfig, ctc_loss
from .text import ALPHABET, ErrorRates, encode, greedy_decode, prefix_beam_search


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name):
    if name == "auto":
        # CPU is the portable fallback on Apple Silicon; MPS CTC/Conv3D
        # support varies between releases and is deliberately not assumed.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    return device






def make_loader(dataset, batch_size, workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_batch,
                      worker_init_fn=seed_worker)


@torch.no_grad()
def evaluate(model, loader, device, beam_width=1, include_predictions=False):
    model.eval()
    metrics, loss_sum, count, predictions = ErrorRates(), 0.0, 0, []
    for batch in loader:
        logits = model(batch["videos"].to(device), batch["lengths"])
        loss = ctc_loss(logits, batch["targets"].to(device), batch["lengths"], batch["target_lengths"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        n = len(batch["texts"])
        loss_sum += float(loss) * n
        count += n
        probs = logits.log_softmax(-1).cpu()
        for i, text in enumerate(batch["texts"]):
            sequence = probs[:int(batch["lengths"][i]), i]
            hypothesis = (greedy_decode(sequence) if beam_width == 1 else
                          prefix_beam_search(sequence, beam_width))
            metrics.update(text, hypothesis)
            if include_predictions:
                predictions.append({"reference": text, "prediction": hypothesis})
    result = {"loss": loss_sum / max(count, 1), **metrics.compute(), "beam_width": beam_width}
    if include_predictions:
        result["predictions"] = predictions
    return result


def train(args, preprocess):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("Epochs, batch size, learning rate and gradient clipping must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start, best = 0, float("inf")
    model = LipNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_data = VideoDataset(args.train, preprocess, training=not args.no_augment)
    val_data = VideoDataset(args.val, preprocess)
    overlap = {r["video"] for r in train_data.rows} & {r["video"] for r in val_data.rows}
    if overlap:
        raise ValueError(f"Training/validation video overlap ({len(overlap)} clips)")
    train_loader = make_loader(train_data, args.batch_size, args.workers, True)
    val_loader = make_loader(val_data, args.batch_size, args.workers)
    train_args = vars(args).copy()
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "training_clips": len(train_data), "validation_clips": len(val_data)}), flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["videos"].to(device), batch["lengths"])
            loss = ctc_loss(logits, batch["targets"].to(device), batch["lengths"], batch["target_lengths"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CTC loss; inspect inputs and learning rate")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            n = len(batch["texts"])
            total += float(loss.detach()) * n
            count += n
            if step % 20 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "train_loss": total / count}), flush=True)
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": total / count, "validation": validation}
        print(json.dumps(record), flush=True)
        improved = validation["loss"] < best
        best = min(best, validation["loss"])
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    return record


