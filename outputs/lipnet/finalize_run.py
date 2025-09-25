"""Verify/export a completed full_train.py run; optionally wait for it to finish."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import zipfile

import torch
from torch.utils.data import DataLoader

from full_train import CachedGrid, evaluate, write_json
from lipnet.data import collate_batch
from lipnet.engine import load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=Path("runs/full-training"))
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    finalization_lock=(args.run/".finalization.lock").open("w")
    try:
        fcntl.flock(finalization_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another finalizer is already monitoring this run",flush=True)
        return
    result_path=args.run/"training-result.json"
    while not result_path.exists():
        progress_path=args.run/"progress.json"
        if progress_path.exists():
            progress=json.loads(progress_path.read_text())
            if progress.get("state")=="failed":
                raise RuntimeError(progress.get("error","Training failed"))
        if not args.wait:raise RuntimeError("Training has not finished")
        time.sleep(30)
    result=json.loads(result_path.read_text())
    if result.get("partial_run"):raise ValueError("Cannot finalize a partial smoke test")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    model,preprocess,state=load_checkpoint(args.run/"best.pt",device)
    manifest=json.loads((args.cache/"manifest.json").read_text())
    dataset=CachedGrid(manifest["validation"],args.cache,preprocess.mean,preprocess.std)
    loader=DataLoader(dataset,batch_size=12,collate_fn=collate_batch)
    evaluated=evaluate(model,loader,device,device.type=="mps")
    original=json.loads((args.run/"best-validation.json").read_text())
    differences=sum(a["prediction"]!=b["prediction"] for a,b in zip(evaluated["predictions"],original["predictions"]))
    evaluated.update({"checkpoint_epoch":state["epoch"],"initialization":"random",
        "evaluation_device":str(device),"reload_prediction_differences":differences,
        "split":"validation used for checkpoint selection, not independent test data",
        "caveat":result["caveat"]})
    write_json(args.run/"final-evaluation.json",evaluated)
    if abs(evaluated["wer"]-original["wer"])>.02:
        raise RuntimeError("Reloaded model WER differs materially; inspect final-evaluation.json")
    # Portable inference artifact without optimizer tensors or absolute dataset paths.
    exported={k:state[k] for k in ("format_version","alphabet","model_config","preprocess","epoch","best_loss","best_wer")}
    exported["model"]={k:v.detach().cpu() for k,v in model.state_dict().items()}
    exported["configuration"]={k:v for k,v in state["configuration"].items() if not k.endswith("_files")}
    temporary=args.run/"model.pt.tmp";torch.save(exported,temporary);temporary.replace(args.run/"model.pt")
    verdict="MET" if result["quality_gate_passed"] else "NOT MET"
    report=f"""# Local LipNet training result

Training completed {result['completed_epochs']} full epochs from random initialization.
The validation quality target was **{verdict}**.

- Data: 900 GRID s1 training clips, 100 validation clips.
- Architecture: {state['configuration']['architecture']}.
- Best checkpoint: epoch {state['epoch']}.
- Reloaded validation word error: {evaluated['wer']:.2%}.
- Reloaded validation character error: {evaluated['cer']:.2%}.
- Exact validation sentences: {evaluated['sentence_accuracy']:.2%}.
- Summed epoch time: {sum(r['seconds'] for r in result['history'])/3600:.2f} hours.
- Reloaded prediction differences: {differences} of {evaluated['clips']} clips.
- Inference weights: `model.pt`; resumable training state: `last.pt` and `best.pt`.

Validation selected the checkpoint. This is not an independent test score or the
paper's benchmark. The data contains one speaker and scripted GRID sentences;
these results do not demonstrate general English or unseen-speaker accuracy.
The upload/webcam app uses separately credited pretrained Auto-AVSR weights.
"""
    (args.run/"TRAINING_RESULT.md").write_text(report)
    project=Path(__file__).resolve().parent
    archive=project.parent/"lipnet-from-scratch.zip"
    temporary=archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
        for path in project.rglob("*"):
            if path.is_file() and not any(p in {"__pycache__",".pytest_cache",".venv"} for p in path.parts) and not path.name.startswith(".") and not path.name.endswith(".tmp"):
                bundle.write(path,path.relative_to(project.parent))
    with zipfile.ZipFile(temporary) as bundle:
        bad=bundle.testzip()
        if bad:raise RuntimeError(f"Archive verification failed: {bad}")
    temporary.replace(archive)
    write_json(args.run/"delivery-status.json",{"state":"verified_and_packaged",
        "quality_gate_passed":result["quality_gate_passed"],"archive":archive.name,
        "checkpoint_epoch":state["epoch"],"wer":evaluated["wer"],"cer":evaluated["cer"]})
    print(report,flush=True)


if __name__=="__main__":main()
