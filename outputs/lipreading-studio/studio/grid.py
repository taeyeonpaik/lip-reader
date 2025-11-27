"""Inference with our separately trained, limited-vocabulary LipNet checkpoint."""
from pathlib import Path
import sys
import json

import torch

PAPER_ROOT = Path(__file__).resolve().parents[2] / "lipnet"


def active_run():
    runs=PAPER_ROOT/"runs"
    pointer=runs/"active-run.json"
    name=json.loads(pointer.read_text())["run"] if pointer.exists() else "full-training"
    path=(runs/name).resolve()
    if path.parent!=runs.resolve():raise ValueError("Invalid training run name")
    return path


def predict_grid(video, checkpoint=None):
    if str(PAPER_ROOT) not in sys.path:
        sys.path.insert(0,str(PAPER_ROOT))
    from lipnet.engine import load_checkpoint
    from lipnet.data import load_video
    from lipnet.text import greedy_decode
    run = active_run()
    checkpoint = Path(checkpoint) if checkpoint else run/("model.pt" if (run/"delivery-status.json").exists() else "best.pt")
    # CPU inference leaves the Mac GPU available for training/live recognition.
    model,preprocess,state = load_checkpoint(checkpoint,torch.device("cpu"))
    model.eval()
    frames = load_video(video,preprocess)
    with torch.inference_mode():
        logits = model(frames.unsqueeze(0),torch.tensor([frames.shape[1]]))
    return {"text":greedy_decode(logits[:,0]),"checkpoint_epoch":state["epoch"],
            "validation_wer":state["best_wer"],"initialization":state["configuration"]["initialization"]}
