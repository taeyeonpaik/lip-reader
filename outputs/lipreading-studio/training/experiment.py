"""Bounded, resumable output-layer adaptation on real transcribed video crops.

The original checkpoint is never overwritten. Validation selects the candidate;
the official test split is evaluated only after that selection is fixed.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import sentencepiece as spm
import torch

from studio.pipeline import LipReader, ROOT, VENDOR
from studio.adaptation import BASE_SHA256, HEAD_NAMES, head_state, set_heads, save_adapter, load_adapter
from training.data import atomic_json, validate_splits
from training.metrics import summarize
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Experiment:
    def __init__(self, args):
        self.args = args
        self.run = args.run.resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.cache = args.cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(args.manifest.read_text())
        validate_splits(self.rows)
        self.splits = {s: [r for r in self.rows if r["split"] == s] for s in ("train", "validation", "test")}
        if any(not rows for rows in self.splits.values()):
            raise ValueError("Training, validation and test splits must all be nonempty")
        with (ROOT / "models/autoavsr.pth").open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != BASE_SHA256:
                raise ValueError("Base checkpoint does not match the recorded author weights")
        self.reader = LipReader(device=args.device, beam_size=args.beam_size)
        self.model = self.reader.model
        self.original = head_state(self.model)
        for name, p in self.model.named_parameters():
            p.requires_grad_(name in HEAD_NAMES)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=args.learning_rate, weight_decay=.01)
        self.sp = spm.SentencePieceProcessor(model_file=str(VENDOR / "spm/unigram/unigram5000.model"))
        self.units = {s.split()[0]: int(s.split()[-1]) for s in (VENDOR / "spm/unigram/unigram5000_units.txt").read_text().splitlines()}
        self.started = time.monotonic()
        self.config = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                       "base_sha256": BASE_SHA256, "learning_rate": args.learning_rate,
                       "epochs": args.epochs, "patience": args.patience, "seed": args.seed,
                       "accumulation": args.accumulation, "beam_size": args.beam_size,
                       "trainable_parameters": sum(p.numel() for p in self.parameters),
                       "trainable_modules": list(HEAD_NAMES), "encoder_device": self.reader.device,
                       "heads_device": "cpu", "splits": {s: len(v) for s,v in self.splits.items()},
                       "minimum_validation_wer_improvement": .005,
                       "input": "Distributor's 96x96 mouth crops; app-equivalent center crop and normalization",
                       "training_mode": "Frozen encoder and decoder body, evaluation-mode dropout, output-head updates only",
                       "test_used_for_selection": False}
        config_path = self.run / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != self.config:
            raise ValueError("Resume configuration differs from the original run")
        atomic_json(config_path, self.config)

    def status(self, stage, **values):
        payload = {"state": "running", "stage": stage, "updated_utc": utc(),
                   "elapsed_seconds": time.monotonic()-self.started, **values}
        atomic_json(self.run / "progress.json", payload)
        print(json.dumps(payload), flush=True)

    def features(self, row):
        # Cache is keyed by video, preprocessing version and exact base checkpoint.
        key = hashlib.sha256((row["sha256"] + BASE_SHA256 + ":rgb-center88-v1").encode()).hexdigest()
        path = self.cache / (key + ".npy")
        if path.exists():
            return torch.from_numpy(np.load(path, allow_pickle=False))
        cap = cv2.VideoCapture(row["path"])
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                if frame.shape != (96,96,3):
                    raise ValueError("Dataset file is not an aligned 96×96 mouth crop")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames or len(frames) != row["declared_frames"] or abs(fps-25) > .01:
            raise ValueError("Dataset frame count or frame rate does not match its manifest")
        # clone outside inference_mode allows output-head autograd on cached features.
        features = self.reader.encode(np.stack(frames)).clone()
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f, features.numpy(), allow_pickle=False)
        tmp.replace(path)
        return features

    def prepare(self, split):
        failures = []
        for index, row in enumerate(self.splits[split], 1):
            try: self.features(row)
            except Exception as error:
                failures.append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
            if index % 25 == 0 or index == len(self.splits[split]):
                self.status("encoding_" + split, completed=index, total=len(self.splits[split]), failures=len(failures))
        atomic_json(self.run / (split + "-preprocessing-failures.json"), failures)
        return {r["id"] for r in failures}

    def evaluate(self, split, label):
        path = self.run / (label + ".json")
        if path.exists():
            return json.loads(path.read_text())["metrics"]
        partial = self.run / (label + ".jsonl")
        records = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
        completed = {r["id"] for r in records}
        with partial.open("a") as f:
            for row in self.splits[split]:
                if row["id"] in completed: continue
                tick = time.monotonic()
                result = {"id": row["id"], "reference": row["text"], "prediction": ""}
                try:
                    features = self.features(row)
                    result["prediction"] = self.reader.decode(features)
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                result["seconds"] = time.monotonic()-tick
                records.append(result)
                f.write(json.dumps(result)+"\n"); f.flush()
                if len(records) % 25 == 0 or len(records) == len(self.splits[split]):
                    self.status(label, completed=len(records), total=len(self.splits[split]))
        metrics = summarize(records)
        metrics["decode_seconds_including_uncached_encoding"] = sum(r["seconds"] for r in records)
        atomic_json(path, {"label": label, "split": split, "beam_size": self.args.beam_size,
                           "input": "Prepared mouth crops, face detector not evaluated",
                           "metrics": metrics, "predictions": records})
        return metrics

    def loss(self, row):
        x = self.features(row).unsqueeze(0)
        tokens = [self.units.get(piece, self.units["<unk>"]) for piece in self.sp.encode(row["text"], out_type=str)]
        if len(tokens) + sum(a==b for a,b in zip(tokens,tokens[1:])) > x.shape[1]:
            raise ValueError("Reference cannot align to these CTC frames")
        labels = torch.tensor([tokens], dtype=torch.long)
        lengths = torch.tensor([x.shape[1]])
        mask = torch.ones(1,1,x.shape[1],dtype=torch.bool)
        in_labels, out_labels = add_sos_eos(labels, self.model.sos, self.model.eos, self.model.ignore_id)
        ctc, _ = self.model.ctc(x, lengths, labels)
        pred, _ = self.model.decoder(in_labels, target_mask(in_labels,self.model.ignore_id), x, mask)
        attention = self.model.criterion(pred, out_labels)
        return .1*ctc + .9*attention

    def run_training(self):
        if (self.run / "result.json").exists():
            print("This experiment is already complete; use a new run directory for a new experiment.", flush=True)
            return
        random.seed(self.args.seed); np.random.seed(self.args.seed); torch.manual_seed(self.args.seed)
        baseline = self.evaluate("validation", "baseline-validation")
        bad = self.prepare("train")
        training = [r for r in self.splits["train"] if r["id"] not in bad]
        if len(training) < .95*len(self.splits["train"]):
            raise ValueError("More than 5% of training videos failed preprocessing")
        history_path = self.run / "history.json"
        history = []
        first_epoch = 1
        best_wer = baseline["wer"]
        best_epoch = 0
        checkpoint_path = self.run / "last.pt"
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            set_heads(self.model, saved["heads"])
            self.optimizer.load_state_dict(saved["optimizer"])
            history = saved["history"]
            first_epoch = saved["epoch"]+1
            best_wer, best_epoch = saved["best_wer"], saved["best_epoch"]
        for epoch in range(first_epoch, self.args.epochs+1):
            # A restart during final test evaluation must not extend an already
            # early-stopped training run by another epoch.
            if epoch-1-best_epoch >= self.args.patience:
                break
            order = list(training)
            random.Random(self.args.seed+epoch).shuffle(order)
            total_loss = 0.
            tick = time.monotonic()
            for begin in range(0,len(order),self.args.accumulation):
                group = order[begin:begin+self.args.accumulation]
                self.optimizer.zero_grad(set_to_none=True)
                for row in group:
                    loss = self.loss(row)
                    if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
                    total_loss += loss.item()
                    (loss/len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
                self.optimizer.step()
                if begin % (self.args.accumulation*25) == 0:
                    self.status("training", epoch=epoch, epochs=self.args.epochs,
                                completed=min(begin+len(group),len(order)), total=len(order), gradient_norm=float(norm))
            # Persist heads before evaluation so interruption does not discard a full epoch.
            save_adapter(self.model, self.run / f"epoch-{epoch:02}.pt", {"epoch":epoch,"training_only":True})
            validation = self.evaluate("validation", f"epoch-{epoch:02}-validation")
            row = {"epoch":epoch,"training_loss":total_loss/len(order),"training_clips":len(order),
                   "seconds":time.monotonic()-tick,"validation":validation}
            history.append(row)
            if validation["wer"] < best_wer and validation["cer"] <= baseline["cer"]:
                best_wer, best_epoch = validation["wer"], epoch
                save_adapter(self.model, self.run / "best-adapter.pt", {"epoch":epoch,"validation":validation})
            payload = {"epoch":epoch,"heads":head_state(self.model),"optimizer":self.optimizer.state_dict(),
                       "history":history,"best_wer":best_wer,"best_epoch":best_epoch}
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload,temporary); temporary.replace(checkpoint_path)
            atomic_json(history_path,history)
            if epoch-best_epoch >= self.args.patience:
                break
        selected = best_epoch > 0 and baseline["wer"]-best_wer >= .005
        selection = {"selected":"adapted" if selected else "pretrained", "best_epoch":best_epoch,
                     "baseline_validation":baseline, "best_validation_wer":best_wer,
                     "required_absolute_wer_gain":.005,"test_used_for_selection":False,
                     "reason":"Validation improvement met the preset threshold" if selected else "Adaptation did not meet the preset validation improvement threshold"}
        atomic_json(self.run / "selection.json",selection)
        # Selection is now fixed. Test references/predictions do not tune any parameters.
        set_heads(self.model,self.original)
        test_baseline = self.evaluate("test", "baseline-test")
        test_selected = test_baseline
        if selected:
            load_adapter(self.model,self.run / "best-adapter.pt")
            test_selected = self.evaluate("test", "selected-test")
        report = {"state":"completed","completed_utc":utc(),"config":self.config,
                  "epochs_completed":len(history),"training_steps":sum((h["training_clips"]+self.args.accumulation-1)//self.args.accumulation for h in history),
                  "real_fine_tuning_completed":True,"selection":selection,
                  "baseline_test":test_baseline,"selected_test":test_selected,
                  "held_out_test_improvement":bool(selected and test_selected["wer"]<test_baseline["wer"]),
                  "production_weights_automatically_changed":False,
                  "limits":["Prepared-mouth LRS2 benchmark, not arbitrary webcam accuracy", "Base model already trained on LRS2", "No unseen-speaker claim", "Small adaptation subset and single training seed"]}
        atomic_json(self.run / "result.json",report)
        self.status("complete",state="completed",epochs_completed=len(history),selected=selection["selected"])
        print(json.dumps(report,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--device",default="auto",choices=["auto","mps","cpu","cuda"])
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--patience",type=int,default=3)
    p.add_argument("--learning-rate",type=float,default=3e-6)
    p.add_argument("--accumulation",type=int,default=8)
    p.add_argument("--beam-size",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    args = p.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    with (args.run / ".lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            experiment = Experiment(args)
            experiment.run_training()
        except BaseException as error:
            atomic_json(args.run / "progress.json", {"state":"failed","updated_utc":utc(),"error":f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    main()
