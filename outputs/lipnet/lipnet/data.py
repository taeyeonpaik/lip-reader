"""Video-only loading, GRID alignments, deterministic splits and augmentation."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .text import encode, normalize


@dataclass
class PreprocessConfig:
    # x, y, width, height in source-frame pixels; None means already cropped.
    crop: tuple | None = None
    fps: float = 25.0
    mean: tuple = (0.7136, 0.4906, 0.3283)
    std: tuple = (0.1138, 0.1078, 0.0917)
    max_frames: int = 300

    def __post_init__(self):
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise ValueError("Provide three RGB means and positive standard deviations")
        if self.fps <= 0 or self.max_frames < 1:
            raise ValueError("fps and max_frames must be positive")
        if self.crop is not None and (len(self.crop) != 4 or min(self.crop[:2]) < 0 or min(self.crop[2:]) <= 0):
            raise ValueError("Crop must be x,y,width,height with positive dimensions")

    def to_dict(self):
        return asdict(self)


def load_video(path, config, standardize=True):
    """Read RGB mouth frames; reject wrong FPS rather than silently change timing.

    No audio is read. Full-frame input is valid only for already cropped clips.
    Returns float32 [3,T,50,100]. Oversized videos fail rather than truncate labels.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or abs(fps - config.fps) > 0.5:
            raise ValueError(f"{path}: expected {config.fps} fps, got {fps}; resample beforehand")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if config.crop is not None:
                x, y, w, h = config.crop
                if x + w > frame.shape[1] or y + h > frame.shape[0]:
                    raise ValueError(f"Crop {config.crop} exceeds frame dimensions in {path}")
                frame = frame[y:y + h, x:x + w]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (100, 50), interpolation=cv2.INTER_AREA))
            if len(frames) > config.max_frames:
                raise ValueError(f"{path}: exceeds max_frames={config.max_frames}; segment it beforehand")
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if reported_count > 0 and len(frames) + 1 < reported_count:
            raise ValueError(f"{path}: decoded {len(frames)} of {reported_count:g} reported frames")
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decodable frames: {path}")
    data = np.stack(frames).astype(np.float32) / 255.0
    if standardize:
        data = (data - np.asarray(config.mean, np.float32)) / np.asarray(config.std, np.float32)
    return torch.from_numpy(data.transpose(3, 0, 1, 2).copy())


def read_alignment(path):
    words, previous_end = [], 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid alignment line in {path}: {line!r}")
        start, end = map(int, parts[:2])
        if start < previous_end or end <= start:
            raise ValueError(f"Unordered/invalid alignment timing in {path}: {line!r}")
        previous_end = end
        if parts[2].lower() not in {"sil", "sp"}:
            words.append(parts[2])
    text = normalize(" ".join(words))
    encode(text)
    return text


def read_manifest(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        video = Path(row["video"])
        if not video.is_absolute():
            video = path.parent / video
        video = video.resolve()
        if not video.is_file():
            raise ValueError(f"{path}:{number}: missing video {video}")
        if str(video) in seen:
            raise ValueError(f"{path}:{number}: duplicate video {video}")
        seen.add(str(video))
        row["video"] = str(video)
        row["text"] = normalize(row["text"])
        encode(row["text"])
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def speaker_of(path):
    for part in reversed(Path(path).parts[:-1]):
        if re.fullmatch(r"s\d+", part.lower()):
            return part.lower()
    raise ValueError(f"Path must contain a speaker directory such as s1: {path}")


def prepare_grid(video_root, alignment_root, output, split="unseen", seed=42, heldout=255):
    """Match by (speaker, utterance), never filename alone across speakers."""
    alignments = {}
    for path in sorted(Path(alignment_root).rglob("*.align")):
        key = (speaker_of(path), path.stem)
        if key in alignments:
            raise ValueError(f"Duplicate alignment key {key}")
        alignments[key] = path
    grouped, seen = {}, set()
    for path in sorted(Path(video_root).rglob("*")):
        if path.suffix.lower() not in {".mpg", ".mpeg", ".mp4", ".avi", ".mov"}:
            continue
        speaker = speaker_of(path)
        key = (speaker, path.stem)
        if key in seen:
            raise ValueError(f"Duplicate video key {key}")
        seen.add(key)
        if key not in alignments:
            raise ValueError(f"Missing alignment for {path}")
        grouped.setdefault(speaker, []).append({"video": str(path.resolve()),
            "text": read_alignment(alignments[key]), "speaker": speaker})
    rng = random.Random(seed)
    train, evaluation = [], []
    for speaker, rows in sorted(grouped.items()):
        if split == "unseen":
            (evaluation if speaker in {"s1", "s2", "s20", "s22"} else train).extend(rows)
        elif split == "overlapped":
            if heldout < 1 or len(rows) <= heldout:
                raise ValueError(f"{speaker}: need more than {heldout} clips for overlapped split")
            rng.shuffle(rows)
            evaluation.extend(rows[:heldout])
            train.extend(rows[heldout:])
        else:
            raise ValueError(f"Unknown split: {split}")
    if not train or not evaluation:
        raise ValueError("Split produced empty training or evaluation data; check speakers and paths")
    output = Path(output)
    write_manifest(train, output / "train.jsonl")
    write_manifest(evaluation, output / "eval.jsonl")
    metadata = {"split": split, "seed": seed, "train": len(train), "eval": len(evaluation),
                "heldout_per_speaker": heldout if split == "overlapped" else None}
    (output / "split.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def augment(video, labels):
    if random.random() < 0.5:
        video = video.flip(-1)
    # Independent 5% chance of deleting or duplicating a frame (2.5% each).
    indices = []
    for t in range(video.shape[1]):
        p = random.random()
        if p < 0.025:
            continue
        indices.append(t)
        if p > 0.975:
            indices.append(t)
    minimum = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
    if len(indices) >= minimum:
        video = video[:, indices]
    return video


class VideoDataset(Dataset):
    def __init__(self, manifest, config, training=False):
        self.rows = read_manifest(manifest)
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labels = encode(row["text"])
        video = load_video(row["video"], self.config)
        if self.training:
            video = augment(video, labels)
        return video, torch.tensor(labels, dtype=torch.long), row["text"]


def collate_batch(samples):
    lengths = torch.tensor([v.shape[1] for v, _, _ in samples], dtype=torch.long)
    videos = samples[0][0].new_zeros(len(samples), 3, int(lengths.max()), 50, 100)
    for i, (video, _, _) in enumerate(samples):
        videos[i, :, :video.shape[1]] = video
    return {"videos": videos, "lengths": lengths,
            "targets": torch.cat([labels for _, labels, _ in samples]),
            "target_lengths": torch.tensor([len(labels) for _, labels, _ in samples]),
            "texts": [text for _, _, text in samples]}


def channel_statistics(manifest, config):
    """Pixel-weighted streaming statistics on TRAINING data only."""
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for row in read_manifest(manifest):
        frames = load_video(row["video"], config, standardize=False).double().flatten(1)
        total += frames.sum(1)
        squares += frames.square().sum(1)
        count += frames.shape[1]
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "pixels_per_channel": count}
