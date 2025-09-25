"""Command line interface. Run python -m lipnet --help."""

import argparse
import json
from pathlib import Path

import torch

from .data import (PreprocessConfig, VideoDataset, channel_statistics,
                   load_video, prepare_grid)
from .engine import (evaluate, load_checkpoint, make_loader, resolve_device,
                     smoke_test, train)
from .text import greedy_decode, prefix_beam_search


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def preprocessing_options(parser):
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Mouth ROI in source pixels; omit only for already cropped videos")
    parser.add_argument("--stats", help="JSON containing training-set mean/std; default: paper values")


def preprocessing(args):
    values = {"crop": args.crop}
    if args.stats:
        stats = json.loads(Path(args.stats).read_text())
        values.update(mean=stats["mean"], std=stats["std"])
    return PreprocessConfig(**values)


def build_parser():
    parser = argparse.ArgumentParser(description="LipNet from scratch: sentence-level visual speech recognition")
    parser.add_argument("--threads", type=positive_int, default=4, help="PyTorch CPU thread count")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create deterministic GRID train/eval JSONL manifests")
    prepare.add_argument("--videos", required=True)
    prepare.add_argument("--alignments", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", choices=["unseen", "overlapped"], default="unseen")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--heldout", type=positive_int, default=255)
    stats = commands.add_parser("stats", help="Compute RGB statistics from training clips")
    stats.add_argument("--manifest", required=True)
    stats.add_argument("--output", required=True)
    preprocessing_options(stats)
    check = commands.add_parser("check", help="Decode every clip and check CTC alignability before training")
    check.add_argument("--manifest", required=True)
    preprocessing_options(check)
    fit = commands.add_parser("train", help="Train from random weights or resume a checkpoint")
    fit.add_argument("--train", required=True)
    fit.add_argument("--val", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=positive_int, default=100)
    fit.add_argument("--batch-size", type=positive_int, default=4)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--grad-clip", type=float, default=5.0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--resume", help="Restores model, optimizer, epoch, RNG and preprocessing")
    fit.add_argument("--no-augment", action="store_true")
    preprocessing_options(fit)
    evaluate_parser = commands.add_parser("evaluate", help="Measure CER/WER on a held-out manifest")
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--batch-size", type=positive_int, default=4)
    evaluate_parser.add_argument("--workers", type=int, default=0)
    evaluate_parser.add_argument("--beam-width", type=positive_int, default=200)
    evaluate_parser.add_argument("--output", help="Save metrics and reference/prediction pairs as JSON")
    predict = commands.add_parser("predict", help="Transcribe a video using a trained checkpoint")
    predict.add_argument("--video", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--beam-width", type=positive_int, default=200)
    smoke = commands.add_parser("smoke", help="Train on artificial video patterns to verify the pipeline")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--steps", type=positive_int, default=80)
    smoke.add_argument("--full-model", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if hasattr(args, "workers") and args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.command == "prepare":
        result = prepare_grid(args.videos, args.alignments, args.output, args.split, args.seed, args.heldout)
    elif args.command == "stats":
        result = channel_statistics(args.manifest, preprocessing(args))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "check":
        data = VideoDataset(args.manifest, preprocessing(args))
        for index in range(len(data)):
            video, labels, _ = data[index]
            needed = len(labels) + int((labels[1:] == labels[:-1]).sum())
            if video.shape[1] < needed:
                raise ValueError(f"{data.rows[index]['video']}: CTC needs {needed} frames")
        result = {"validated_clips": len(data)}
    elif args.command == "train":
        if args.resume and (args.crop is not None or args.stats):
            parser.error("Resume restores preprocessing; omit --crop and --stats")
        result = train(args, preprocessing(args))
    elif args.command in {"evaluate", "predict"}:
        device = resolve_device(args.device)
        model, config, _ = load_checkpoint(args.checkpoint, device)
        model.eval()
        if args.command == "evaluate":
            data = VideoDataset(args.manifest, config)
            loader = make_loader(data, args.batch_size, args.workers)
            result = evaluate(model, loader, device, args.beam_width, include_predictions=bool(args.output))
            if args.output:
                for row, prediction in zip(data.rows, result["predictions"]):
                    prediction["video"] = row["video"]
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n")
        else:
            video = load_video(args.video, config)
            with torch.inference_mode():
                probs = model(video.unsqueeze(0).to(device), [video.shape[1]]).log_softmax(-1)[:, 0].cpu()
            text = greedy_decode(probs) if args.beam_width == 1 else prefix_beam_search(probs, args.beam_width)
            result = {"video": args.video, "text": text, "frames": video.shape[1]}
    else:
        result = smoke_test(args.output, args.steps, args.full_model)
    print(json.dumps(result, indent=2))
