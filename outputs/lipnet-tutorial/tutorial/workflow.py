"""Dataset partitioning, training callbacks, evaluation and visual demonstration."""

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
import tf_keras as keras

from .core import (VIDEO_SHAPE, alignment_text, build_model, ctc_loss, decode,
                   encode, learning_rate, load_model, read_video, tutorial_ctc_loss)


def discover(data):
    data = Path(data).resolve()
    rows = []
    for video in sorted((data / "s1").glob("*.mpg")):
        alignment = data / "alignments" / "s1" / (video.stem + ".align")
        if not alignment.is_file():
            raise ValueError(f"Missing alignment: {alignment}")
        text = alignment_text(alignment)
        if len(encode(text)) > 40:
            raise ValueError(f"Transcript exceeds tutorial's 40 character limit: {video}")
        rows.append({"video": str(video), "alignment": str(alignment), "text": text})
    if len(rows) < 2:
        raise ValueError("Expected data/s1/*.mpg and data/alignments/s1/*.align")
    return rows


def split_data(data, seed=42):
    rows = discover(data)
    random.Random(seed).shuffle(rows)
    # 450 batches of two gives 900 training clips in the 1000-clip download.
    cutoff = 900 if len(rows) == 1000 else max(1, int(0.9 * len(rows)))
    return rows[:cutoff], rows[cutoff:]


def dataset(rows, batch_size=2, shuffle=False, seed=42):
    paths = [row["video"] for row in rows]
    texts = [row["text"] for row in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, texts))
    if shuffle:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=False)

    def load(path, text):
        def python_load(p, t):
            frames, _ = read_video(p.numpy().decode())
            target = encode(t.numpy().decode())
            return frames, target
        frames, target = tf.py_function(python_load, [path, text], [tf.float32, tf.int64])
        frames.set_shape(VIDEO_SHAPE)
        target.set_shape([None])
        return frames, target

    ds = ds.map(load, num_parallel_calls=1)
    ds = ds.padded_batch(batch_size, padded_shapes=(VIDEO_SHAPE, [40]),
                         padding_values=(tf.constant(0, tf.float32), tf.constant(0, tf.int64)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)


class EpochExamples(keras.callbacks.Callback):
    def __init__(self, validation):
        super().__init__()
        self.validation = validation

    def on_epoch_end(self, epoch, logs=None):
        # Start a fresh iterator each time; the tutorial's iterator can exhaust.
        videos, labels = next(iter(self.validation))
        predictions, _ = decode(self.model(videos, training=False), beam_width=10)
        from .core import tokens_to_text
        for target, prediction in zip(labels.numpy(), predictions):
            print(json.dumps({"epoch": epoch + 1, "reference": tokens_to_text(target),
                              "prediction": prediction}), flush=True)


class SaveState(keras.callbacks.Callback):
    def __init__(self, checkpoint, manager, output):
        super().__init__()
        self.checkpoint, self.manager, self.output = checkpoint, manager, output

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint.epoch.assign(epoch + 1)
        self.manager.save(checkpoint_number=epoch + 1)
        self.model.save_weights(str(self.output / "last.weights.h5"))


class Progress(keras.callbacks.Callback):
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.started = time.monotonic()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 10 == 0:
            record = {"epoch": self.epoch, "batch": batch + 1,
                      "elapsed_seconds": time.monotonic() - self.started,
                      "loss": float(logs["loss"])}
            path = self.output / "progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(path)
            print(json.dumps(record), flush=True)


def train(data, output, epochs=2, batch_size=2, seed=42, steps=None,
          validation_steps=None, resume=False, loss_mode="tutorial"):
    if epochs < 1 or batch_size < 1 or (steps is not None and steps < 1):
        raise ValueError("Epochs, batch size and optional steps must be positive")
    keras.utils.set_random_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run.json"
    train_rows, val_rows = split_data(data, seed)
    configuration = {"data": str(Path(data).resolve()), "seed": seed, "batch_size": batch_size,
                     "loss_mode": loss_mode, "training_clips": len(train_rows),
                     "validation_clips": len(val_rows),
                     "train_files": [row["video"] for row in train_rows],
                     "validation_files": [row["video"] for row in val_rows],
                     "steps_per_epoch": steps, "validation_steps": validation_steps}
    if config_path.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume or another output directory")
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Resume configuration differs from the saved run")
    elif resume:
        raise ValueError("No saved run to resume")
    else:
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    training = dataset(train_rows, batch_size, shuffle=True, seed=seed)
    validation = dataset(val_rows, batch_size)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=tutorial_ctc_loss if loss_mode == "tutorial" else ctc_loss)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer,
                                     epoch=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=2)
    if resume:
        if not manager.latest_checkpoint:
            raise ValueError("No complete epoch checkpoint exists")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    start_epoch = int(checkpoint.epoch.numpy())
    if epochs <= start_epoch:
        raise ValueError(f"--epochs must exceed completed epoch {start_epoch}")
    callbacks = [Progress(output), keras.callbacks.LearningRateScheduler(learning_rate),
                 SaveState(checkpoint, manager, output), EpochExamples(validation),
                 keras.callbacks.CSVLogger(str(output / "history.csv"), append=resume),
                 keras.callbacks.TerminateOnNaN()]
    start = time.monotonic()
    history = model.fit(training, validation_data=validation, initial_epoch=start_epoch,
                        epochs=epochs, steps_per_epoch=steps, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=2)
    result = {"seconds": time.monotonic() - start, "initial_epoch": start_epoch,
              "completed_epoch": int(checkpoint.epoch.numpy()),
              "history": {key: [float(v) for v in values] for key, values in history.history.items()},
              "partial_epochs": steps is not None, "parameters": model.count_params()}
    (output / "training-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def predict(model, path, alignment=None, beam_width=1):
    frames, _ = read_video(path)
    start = time.monotonic()
    probabilities = model(frames[None], training=False)
    texts, tokens = decode(probabilities, beam_width)
    result = {"video": str(path), "prediction": texts[0],
              "tokens": [int(t) for t in tokens[0] if t >= 0],
              "inference_seconds": time.monotonic() - start}
    if alignment:
        result["reference"] = alignment_text(alignment)
        result["exact_match"] = result["reference"] == result["prediction"]
    return result


def evaluate(weights, data, output, limit=None, seed=42, beam_width=1):
    _, rows = split_data(data, seed)
    rows = rows if limit is None else rows[:limit]
    model = load_model(weights)
    results = []
    for i, row in enumerate(rows, 1):
        result = predict(model, row["video"], row["alignment"], beam_width)
        results.append(result)
        print(json.dumps({"clip": i, **result}), flush=True)
    characters = sum(len(r["reference"]) for r in results)
    words = sum(len(r["reference"].split()) for r in results)
    saved_run = Path(weights).parent / "run.json"
    independent = False
    if saved_run.exists():
        configuration = json.loads(saved_run.read_text())
        independent = (configuration.get("seed") == seed and
                       all(row["video"] in configuration.get("validation_files", []) for row in rows))
    report = {"clips": len(results), "weights": str(weights),
              "cer": sum(distance(r["reference"], r["prediction"]) for r in results) / max(characters, 1),
              "wer": sum(distance(r["reference"].split(), r["prediction"].split()) for r in results) / max(words, 1),
              "exact_matches": sum(r["exact_match"] for r in results), "predictions": results,
              "caveat": ("Held-out validation clips from this local run. This is single-speaker GRID validation, not a new-speaker or general-video benchmark."
                         if independent else "The checkpoint's original training split is unknown; these are demonstration metrics, not an independent test benchmark.")}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def create_preview(video, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, crops = read_video(video)
    gif = output / (Path(video).stem + "-mouth.gif")
    imageio.mimsave(gif, crops[..., 0], duration=40, loop=0)
    model_gif = output / (Path(video).stem + "-input.gif")
    display = ((frames - frames.min()) / max(float(np.ptp(frames)), 1e-6) * 255).astype(np.uint8)
    imageio.mimsave(model_gif, display[..., 0], duration=40, loop=0)
    mp4 = output / (Path(video).stem + ".mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    return {"video": str(mp4), "mouth": str(gif), "model_input": str(model_gif)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
