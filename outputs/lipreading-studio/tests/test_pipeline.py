import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio.pipeline import ROOT, frame_chunks, prepare_video, subtitles, track_mouth


def test_audio_is_removed_and_does_not_change_model_pixels(tmp_path):
    original = ROOT / "examples/research-demo.mp4"
    with_audio = tmp_path / "audio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(original), "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=8", "-c:v", "copy", "-c:a", "aac",
                    str(with_audio)], check=True)
    a, b = tmp_path/"a.mp4", tmp_path/"b.mp4"
    assert prepare_video(original, a) == pytest.approx(8, abs=.1)
    prepare_video(with_audio, b)
    streams = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(b)]))["streams"]
    assert len(streams) == 1 and streams[0]["codec_type"] == "video"
    first = list(frame_chunks(a))
    second = list(frame_chunks(b))
    np.testing.assert_array_equal(first[0][1], second[0][1])
    assert len(first[0][1]) == 200


def test_chunks_preserve_tail_and_timestamps(tmp_path):
    file = tmp_path/"frames.avi"
    writer = cv2.VideoWriter(str(file), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64,64))
    for i in range(80):
        writer.write(np.full((64,64,3), i, dtype=np.uint8))
    writer.release()
    chunks = list(frame_chunks(file, seconds=3))
    assert [(start, len(frames)) for start, frames in chunks] == [(0,75),(75,5)]
    assert np.mean(chunks[-1][1][-1]) == pytest.approx(79, abs=2)


def test_invalid_file_fails_before_inference(tmp_path):
    source = tmp_path/"not-a-video.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="readable video"):
        prepare_video(source, tmp_path/"out.mp4")


def test_no_face_is_rejected():
    with pytest.raises(ValueError, match="tracked reliably"):
        track_mouth(np.zeros((15,128,128,3), dtype=np.uint8))


def test_real_face_alignment(tmp_path):
    normalized = tmp_path/"video.mp4"
    prepare_video(ROOT/"examples/research-demo.mp4", normalized)
    _, frames = next(frame_chunks(normalized))
    crops, coverage = track_mouth(frames)
    assert crops.shape == (200,96,96,3)
    assert coverage >= .9
    assert np.std(crops) > 10


def test_subtitle_timestamps_and_partial_segment():
    result = {"segments": [{"start":0,"end":12,"text":"FIRST"},
                            {"start":12,"end":12.44,"text":"SECOND"}]}
    assert subtitles(result) == "1\n00:00:00,000 --> 00:00:12,000\nFIRST\n\n2\n00:00:12,000 --> 00:00:12,440\nSECOND\n"
