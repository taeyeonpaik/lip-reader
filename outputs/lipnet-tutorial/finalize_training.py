"""Update training status and package results when the long local run finishes.

This watches files from an already running job; it never starts or restarts
training. It exits after completion, a logged failure, or stalled progress.
"""

import argparse
import json
import math
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def update_status(run, log=None):
    progress = read_json(run / "progress.json")
    result = read_json(run / "training-result.json")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = "running"
    details = "The full two-epoch CPU run is still in progress."
    if result:
        finite = all(math.isfinite(v) for values in result["history"].values() for v in values)
        if (result["completed_epoch"] == 2 and not result["partial_epochs"] and finite
                and progress and progress["epoch"] == 2 and progress["batch"] == 450):
            state = "complete"
            details = ("Two full epochs completed: 900 training clips and 100 validation clips per epoch. "
                       "These locally trained weights are separate from the author's 96-epoch demo weights.")
        else:
            state = "incomplete"
            details = "Training returned, but the result does not establish two complete finite-loss epochs."
    elif log and log.exists() and "Traceback (most recent call last)" in log.read_text()[-12000:]:
        state = "failed"
        details = "The training process reported an error. Its log needs inspection; completion is not claimed."
    elif progress and time.time() - (run / "progress.json").stat().st_mtime > 1200:
        state = "stalled"
        details = "No new batch progress has been recorded for 20 minutes. The process needs inspection."
    record = {"status": state, "updated_utc": now, "details": details, "progress": progress}
    if result:
        record["result"] = result
    validation = ROOT / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "training-status.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = ["# Local training status", "", f"**{state.upper()}** — updated {now}", "", details, ""]
    if progress:
        lines.extend([f"Last saved progress: epoch {progress['epoch']}/2, batch {progress['batch']}/450.",
                      f"Elapsed at that point: {progress['elapsed_seconds']/60:.1f} minutes.",
                      f"Reported running CTC loss: {progress['loss']:.4f}.", ""])
    if result:
        lines.extend([f"Run duration: {result['seconds']/3600:.2f} hours.", "",
                      "Completed-epoch values:", "", "```json", json.dumps(result["history"], indent=2), "```", ""])
    lines.extend(["The interactive demo continues to use the author's published checkpoint.",
                  "This file is updated by a local status-watcher process while training runs.",
                  "It does not imply that an assistant is continuously reasoning in the background.", ""])
    (ROOT / "TRAINING_STATUS.md").write_text("\n".join(lines))
    if state == "complete":
        shutil.copyfile(run / "training-result.json", validation / "full-training.json")
        shutil.copyfile(run / "history.csv", validation / "full-training-history.csv")
    return state


def package():
    destination = ROOT.parent / "lipnet-video-tutorial.zip"
    with tempfile.NamedTemporaryFile(dir=ROOT.parent, suffix=".zip.tmp", delete=False) as temporary:
        name = Path(temporary.name)
    try:
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(ROOT.rglob("*")):
                if not path.is_file():
                    continue
                parts = path.relative_to(ROOT).parts
                if any(part in {"__pycache__", ".pytest_cache", ".venv", "assets", ".ipynb_checkpoints"} for part in parts):
                    continue
                if path.name.endswith(".tmp"):
                    continue
                archive.write(path, path.relative_to(ROOT.parent))
        name.replace(destination)
    finally:
        name.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "runs/two-epochs"))
    parser.add_argument("--log")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    previous = None
    while True:
        state = update_status(run, Path(args.log) if args.log else None)
        if state != previous:
            print(state, flush=True)
            previous = state
        if not args.watch or state != "running":
            if args.package or state == "complete":
                print(package(), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
