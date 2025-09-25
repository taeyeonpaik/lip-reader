"""Complete, resumable Mac GPU training of the paper architecture on real GRID data."""
import argparse
import fcntl
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lipnet.data import PreprocessConfig, collate_batch, load_video, read_alignment
from lipnet.model import LipNet, ModelConfig
from lipnet.text import ALPHABET, ErrorRates, encode, greedy_decode


def write_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(content,indent=2)+"\n")
    temporary.replace(path)


def cache_data(manifest, cache):
    cache = Path(cache); cache.mkdir(parents=True,exist_ok=True)
    rows = manifest["train"]+manifest["validation"]
    train_ids = {Path(r["video"]).stem for r in manifest["train"]}
    validation_ids = {Path(r["video"]).stem for r in manifest["validation"]}
    if train_ids & validation_ids or len(train_ids|validation_ids)!=len(rows):
        raise ValueError("Duplicate or overlapping clip IDs")
    crop = PreprocessConfig(crop=(80,190,140,46))
    total = np.zeros(3,dtype=np.float64)
    squares = np.zeros(3,dtype=np.float64)
    pixels = 0
    for index,row in enumerate(rows,1):
        identity = Path(row["video"]).stem
        path = cache/(identity+".npy")
        if not path.exists():
            frames = load_video(row["video"],crop,standardize=False)
            rgb = np.rint(frames.numpy().transpose(1,2,3,0)*255).astype(np.uint8)
            np.save(path,rgb)
        if identity in train_ids:
            rgb = np.load(path).reshape(-1,3).astype(np.float64)/255
            total += rgb.sum(0); squares += np.square(rgb).sum(0); pixels += len(rgb)
        if index%100==0:
            print(json.dumps({"preprocessed":index,"total":len(rows)}),flush=True)
    mean = total/pixels
    std = np.sqrt(squares/pixels-mean**2)
    metadata = {"crop":[80,190,140,46],"mean":mean.tolist(),"std":std.tolist(),
                "statistics_source":"training clips only", "train":manifest["train"],
                "validation":manifest["validation"],"rgb_shape":"T,50,100,3",
                "initialization":"random; no pretrained weights"}
    write_json(cache/"manifest.json",metadata)
    return metadata


class CachedGrid(Dataset):
    def __init__(self, rows, cache, mean, std, training=False):
        self.rows, self.cache, self.training = rows,Path(cache),training
        self.mean = np.array(mean,dtype=np.float32)
        self.std = np.array(std,dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        video = np.load(self.cache/(Path(row["video"]).stem+".npy")).astype(np.float32)/255
        video = (video-self.mean)/self.std
        if self.training and random.random()<.5:
            video = video[:,:,::-1,:]
        video = torch.from_numpy(video.transpose(3,0,1,2).copy())
        return video,torch.tensor(encode(row["text"])),row["text"]


def loss_for_batch(logits,batch):
    # CPU CTC with differentiable transfer preserves GPU CNN/GRU gradients.
    return torch.nn.functional.ctc_loss(logits.float().log_softmax(-1).cpu(),
        batch["targets"],batch["lengths"],batch["target_lengths"],blank=0,
        reduction="mean",zero_infinity=False)


@torch.no_grad()
def evaluate(model,loader,device,amp):
    model.eval()
    errors=ErrorRates();count=0;total=0.;predictions=[]
    for batch in loader:
        with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
            logits=model(batch["videos"].to(device),batch["lengths"])
        loss=loss_for_batch(logits,batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total+=loss.item()*len(batch["texts"]);count+=len(batch["texts"])
        logits=logits.float().cpu()
        for index,reference in enumerate(batch["texts"]):
            prediction=greedy_decode(logits[:batch["lengths"][index],index])
            errors.update(reference,prediction)
            predictions.append({"reference":reference,"prediction":prediction})
    return {"loss":total/count,**errors.compute(),"clips":count,
            "unique_predictions":len({row["prediction"] for row in predictions}),"predictions":predictions}


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest",type=Path)
    source.add_argument("--data",type=Path,help="Tutorial dataset directory containing s1 and alignments/s1")
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--batch-size",type=int,default=12)
    parser.add_argument("--lr",type=float,default=.0003)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--feature-norm",action="store_true",help="Normalize CNN features before the GRU to prevent saturation")
    parser.add_argument("--dropout",type=float,default=.5)
    parser.add_argument("--max-steps",type=int,help="Explicitly partial smoke test; never a complete epoch")
    parser.add_argument("--device",choices=["mps","cpu"],default="mps")
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or (args.max_steps is not None and args.max_steps<1):
        parser.error("Training dimensions and learning rate must be positive")
    if not 0<=args.dropout<1:parser.error("Dropout must be in [0,1)")
    args.output.mkdir(parents=True,exist_ok=True)
    run_lock=(args.output/".training.lock").open("w")
    try:
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A training process already owns this output directory") from None
    if (args.output/"last.pt").exists() and not args.resume:
        raise ValueError("Run already exists; specify --resume")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=="mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Launch from a normal Terminal or use --device cpu.")
    amp=device.type=="mps"
    if amp:
        torch.mps.manual_seed(args.seed)
        torch.mps.set_per_process_memory_fraction(.65)
    cache_manifest=args.cache/"manifest.json"
    if args.manifest:
        source_manifest=json.loads(args.manifest.read_text())
    else:
        rows=[]
        for path in sorted((args.data/"s1").glob("*.mpg")):
            alignment=args.data/"alignments/s1"/(path.stem+".align")
            rows.append({"video":str(path.resolve()),"alignment":str(alignment.resolve()),"text":read_alignment(alignment)})
        if len(rows)<10:raise ValueError("Expected GRID data/s1 videos and matching alignments")
        random.Random(args.seed).shuffle(rows)
        cutoff=int(.9*len(rows));source_manifest={"train":rows[:cutoff],"validation":rows[cutoff:]}
    data=json.loads(cache_manifest.read_text()) if cache_manifest.exists() else cache_data(source_manifest,args.cache)
    for split in ("train","validation"):
        if data[split]!=source_manifest[split]:raise ValueError("Cache belongs to a different dataset or split; choose a new cache directory")
    training=CachedGrid(data["train"],args.cache,data["mean"],data["std"],training=True)
    validation=CachedGrid(data["validation"],args.cache,data["mean"],data["std"])
    train_loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,collate_fn=collate_batch)
    val_loader=DataLoader(validation,batch_size=args.batch_size,collate_fn=collate_batch)
    model=LipNet(ModelConfig(feature_norm=args.feature_norm,dropout=args.dropout)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler(device.type,enabled=amp,init_scale=128)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=8,min_lr=1e-6)
    configuration={"architecture":"LipNet paper CNN + reset-before Bi-GRU"+(" with feature LayerNorm" if args.feature_norm else ""),
        "model_config":model.config.to_dict(),
        "parameters":sum(p.numel() for p in model.parameters()),"initialization":"random",
        "pretrained_weights":None,"seed":args.seed,"batch_size":args.batch_size,
        "initial_lr":args.lr,"max_epochs":args.epochs,"device":str(device),"amp":amp,
        "training_clips":len(training),"validation_clips":len(validation),
        "partial_run":args.max_steps is not None,"preprocess":{k:data[k] for k in ("crop","mean","std")},
        "train_files":[r["video"] for r in data["train"]],
        "validation_files":[r["video"] for r in data["validation"]],
        "quality_gate":{"max_validation_wer":.10,"max_validation_cer":.04,"min_epochs":40,"consecutive_epochs":3},
        "caveat":"Single-speaker GRID validation; does not establish arbitrary-video or new-speaker accuracy."}
    start_epoch=0;best_loss=math.inf;best_wer=math.inf;good_epochs=0;history=[];skipped_updates=0
    if args.resume:
        saved=torch.load(args.output/"last.pt",map_location="cpu",weights_only=True)
        for key in ("architecture","seed","batch_size","preprocess","train_files","validation_files","partial_run"):
            if saved["configuration"][key]!=configuration[key]:
                raise ValueError(f"Resume mismatch: {key}")
        if saved["model_config"]!=model.config.to_dict():
            # Older checkpoints omit feature_norm, which defaults to False.
            if ModelConfig(**saved["model_config"]).to_dict()!=model.config.to_dict():
                raise ValueError("Resume mismatch: model configuration")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"]);scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"]);random.setstate(saved["python_rng"])
        if amp:torch.mps.set_rng_state(saved["mps_rng"])
        start_epoch=saved["epoch"];best_loss=saved["best_loss"];best_wer=saved["best_wer"]
        history=saved["history"];good_epochs=saved["good_epochs"]
        skipped_updates=saved.get("skipped_updates",0)
        if args.epochs<=start_epoch:
            raise ValueError("--epochs must exceed the last completed epoch when resuming")
    write_json(args.output/"run.json",configuration)
    started=time.monotonic()
    def status(state,epoch,batch,**extra):
        record={"state":state,"epoch":epoch,"max_epochs":args.epochs,"batch":batch,
            "batches_per_epoch":len(train_loader),"elapsed_seconds":time.monotonic()-started,
            "updated_utc":datetime.now(timezone.utc).isoformat(),"best_wer":best_wer if math.isfinite(best_wer) else None,
            "skipped_nonfinite_updates":skipped_updates,**extra}
        write_json(args.output/"progress.json",record)
        print(json.dumps(record),flush=True)
    print(json.dumps({k:v for k,v in configuration.items() if not k.endswith("_files")}),flush=True)
    try:
        for epoch in range(start_epoch+1,args.epochs+1):
            model.train();total=0.;count=0;epoch_start=time.monotonic()
            for step,batch in enumerate(train_loader,1):
                while (args.output/"PAUSE").exists():
                    status("paused",epoch,step-1);time.sleep(5)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                    logits=model(batch["videos"].to(device),batch["lengths"])
                loss=loss_for_batch(logits,batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):skipped_updates+=1
                if skipped_updates>20:
                    raise FloatingPointError("Too many non-finite gradient updates")
                scaler.step(optimizer);scaler.update()
                total+=loss.item()*len(batch["texts"]);count+=len(batch["texts"])
                if step==1 or step%10==0:
                    status("training",epoch,step,train_loss=total/count)
                if args.max_steps is not None and step>=args.max_steps:break
            status("validating",epoch,step,train_loss=total/count)
            validation_result=evaluate(model,val_loader,device,amp)
            scheduler.step(validation_result["loss"])
            record={"epoch":epoch,"train_loss":total/count,
                    "validation":{k:v for k,v in validation_result.items() if k!="predictions"},
                    "seconds":time.monotonic()-epoch_start,"lr":optimizer.param_groups[0]["lr"]}
            history.append(record)
            improved=(validation_result["wer"],validation_result["loss"])<(best_wer,best_loss)
            if improved:best_wer=validation_result["wer"];best_loss=validation_result["loss"]
            good_epochs=good_epochs+1 if validation_result["wer"]<=.10 and validation_result["cer"]<=.04 else 0
            saved={"format_version":1,"alphabet":ALPHABET,"model_config":model.config.to_dict(),
                   "preprocess":PreprocessConfig(**configuration["preprocess"]).to_dict(),
                   "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                   "scheduler":scheduler.state_dict(),"epoch":epoch,"best_loss":best_loss,"best_wer":best_wer,
                   "configuration":configuration,"history":history,"good_epochs":good_epochs,
                   "skipped_updates":skipped_updates,
                   "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
                   "mps_rng":torch.mps.get_rng_state() if amp else None}
            temporary=args.output/"last.pt.tmp";torch.save(saved,temporary);temporary.replace(args.output/"last.pt")
            if improved:
                temporary=args.output/"best.pt.tmp";torch.save(saved,temporary);temporary.replace(args.output/"best.pt")
                write_json(args.output/"best-validation.json",validation_result)
            write_json(args.output/"history.json",history)
            print(json.dumps(record),flush=True)
            if epoch>=30 and len(history)>=5 and all(r["validation"].get("unique_predictions",100)<=2 and r["validation"]["wer"]>.8 for r in history[-5:]):
                raise RuntimeError("Model collapsed to at most two predictions for five epochs; stopping for diagnosis instead of wasting the remaining run")
            if epoch>=40 and good_epochs>=3 and args.max_steps is None:break
        qualified=good_epochs>=3 and epoch>=40 and args.max_steps is None
        status("completed" if args.max_steps is None else "partial_test_complete",epoch,step,
               quality_gate_passed=qualified,completed_epochs=epoch)
        write_json(args.output/"training-result.json",{"completed_epochs":epoch,"quality_gate_passed":qualified,
            "partial_run":args.max_steps is not None,"best_wer":best_wer,"best_loss":best_loss,
            "seconds_this_session":time.monotonic()-started,"history":history,
            "caveat":configuration["caveat"]})
    except BaseException as error:
        status("failed",locals().get("epoch",0),locals().get("step",0),error=f"{type(error).__name__}: {error}")
        raise


if __name__=="__main__":
    main()
