import json

import cv2
import numpy as np
import pytest
import torch

from lipnet.data import (PreprocessConfig, channel_statistics, collate_batch,
                         load_video, prepare_grid, read_alignment, read_manifest,
                         write_manifest)


def make_video(path, frames=8, fps=25):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (120, 60))
    assert writer.isOpened(), "MJPG encoder unavailable"
    for _ in range(frames):
        image = np.zeros((60, 120, 3), dtype=np.uint8)
        image[:] = (20, 80, 160)  # BGR
        writer.write(image)
    writer.release()


def test_video_rgb_crop_normalization_and_failures(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)
    cfg = PreprocessConfig(crop=(10, 10, 100, 50))
    raw = load_video(path, cfg, standardize=False)
    assert raw.shape == (3, 8, 50, 100)
    torch.testing.assert_close(raw.mean((1, 2, 3)), torch.tensor([160, 80, 20]) / 255, atol=.015, rtol=0)
    normalized = load_video(path, cfg)
    expected = (raw - torch.tensor(cfg.mean)[:, None, None, None]) / torch.tensor(cfg.std)[:, None, None, None]
    torch.testing.assert_close(normalized, expected)
    with pytest.raises(ValueError, match="exceeds frame"):
        load_video(path, PreprocessConfig(crop=(100, 100, 10, 10)))
    with pytest.raises(ValueError, match="max_frames"):
        load_video(path, PreprocessConfig(max_frames=2))
    wrong_fps = tmp_path / "wrong.avi"
    make_video(wrong_fps, fps=10)
    with pytest.raises(ValueError, match="fps"):
        load_video(wrong_fps, cfg)


def test_alignment(tmp_path):
    path = tmp_path / "sample.align"
    path.write_text("0 10 sil\n10 20 set\n20 30 blue\n30 40 sp\n40 50 now\n")
    assert read_alignment(path) == "set blue now"
    path.write_text("20 10 blue\n")
    with pytest.raises(ValueError):
        read_alignment(path)


def test_speaker_splits_and_duplicate_validation(tmp_path):
    for speaker in ("s1", "s3"):
        for index in range(3):
            video = tmp_path / "videos" / speaker / f"clip{index}.avi"
            make_video(video)
            alignment = tmp_path / "align" / speaker / f"clip{index}.align"
            alignment.parent.mkdir(parents=True, exist_ok=True)
            alignment.write_text("0 100 a\n")
    out = tmp_path / "manifests"
    result = prepare_grid(tmp_path / "videos", tmp_path / "align", out)
    assert result["train"] == result["eval"] == 3
    assert {r["speaker"] for r in read_manifest(out / "train.jsonl")} == {"s3"}
    assert {r["speaker"] for r in read_manifest(out / "eval.jsonl")} == {"s1"}
    prepare_grid(tmp_path / "videos", tmp_path / "align", out, "overlapped", heldout=1)
    first = (out / "train.jsonl").read_text()
    prepare_grid(tmp_path / "videos", tmp_path / "align", out, "overlapped", heldout=1)
    assert (out / "train.jsonl").read_text() == first
    train = read_manifest(out / "train.jsonl")
    evaluation = read_manifest(out / "eval.jsonl")
    assert not {r["video"] for r in train} & {r["video"] for r in evaluation}
    write_manifest([train[0], train[0]], out / "duplicate.jsonl")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(out / "duplicate.jsonl")


def test_statistics_and_collation(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"video": "sample.avi", "text": "ab"}) + "\n")
    stats = channel_statistics(manifest, PreprocessConfig())
    assert stats["pixels_per_channel"] == 8 * 50 * 100
    assert len(stats["std"]) == 3 and min(stats["std"]) > 0
    batch = collate_batch([(torch.ones(3, 5, 50, 100), torch.tensor([1, 2]), "ab"),
                           (torch.ones(3, 3, 50, 100), torch.tensor([1]), "a")])
    assert batch["videos"].shape == (2, 3, 5, 50, 100)
    assert batch["videos"][1, :, 3:].count_nonzero() == 0
    assert batch["targets"].tolist() == [1, 2, 1]
