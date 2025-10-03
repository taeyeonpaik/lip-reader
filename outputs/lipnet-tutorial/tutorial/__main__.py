import argparse
import json
from pathlib import Path

import tensorflow as tf

from .core import load_model
from .workflow import create_preview, evaluate, predict, train


def main():
    parser = argparse.ArgumentParser(description="Run the TensorFlow lip-reading tutorial")
    parser.add_argument("--threads", type=int, default=4)
    subs = parser.add_subparsers(dest="command", required=True)
    fitting = subs.add_parser("train")
    fitting.add_argument("--data", required=True)
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--epochs", type=int, default=2)
    fitting.add_argument("--batch-size", type=int, default=2)
    fitting.add_argument("--steps", type=int)
    fitting.add_argument("--validation-steps", type=int)
    fitting.add_argument("--resume", action="store_true")
    fitting.add_argument("--loss-mode", choices=["tutorial", "actual-lengths"], default="tutorial")
    fitting.add_argument("--seed", type=int, default=42)
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--weights", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--beam-width", type=int, default=1)
    prediction = subs.add_parser("predict")
    prediction.add_argument("--video", required=True)
    prediction.add_argument("--weights", required=True)
    prediction.add_argument("--alignment")
    prediction.add_argument("--beam-width", type=int, default=1)
    prediction.add_argument("--output")
    preview = subs.add_parser("preview")
    preview.add_argument("--video", required=True)
    preview.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    kwargs = vars(args).copy()
    del kwargs["threads"]
    command = kwargs.pop("command")
    if command == "train":
        result = train(**kwargs)
    elif command == "evaluate":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        result = evaluate(**kwargs)
    elif command == "preview":
        result = create_preview(args.video, args.output)
    else:
        result = predict(load_model(args.weights), args.video, args.alignment, args.beam_width)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
