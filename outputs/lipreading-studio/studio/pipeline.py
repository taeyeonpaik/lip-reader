"""Local visual-only speech recognition for videos outside GRID."""


from pathlib import Path


import json


import subprocess


import sys


import threading


import time


import cv2


import numpy as np


import torch


ROOT = Path(__file__).resolve().parents[1]


VENDOR = ROOT / "vendor" / "auto_avsr"


sys.path.insert(0, str(VENDOR))


from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E


from espnet.nets.batch_beam_search import BatchBeamSearch


from espnet.nets.scorers.length_bonus import LengthBonus


from preparation.detectors.mediapipe.video_process import VideoProcess


def prepare_video(source, destination, max_seconds=60):
    """Normalize rotation, frame rate and size; discard audio before inference."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(source)], capture_output=True, timeout=20)
    if probe.returncode:
        raise ValueError("This file is not a readable video. Try an MP4 or MOV file.")
    info = json.loads(probe.stdout)
    video = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("The file has no video track.")
    duration = float(video.get("duration", info.get("format", {}).get("duration", 0)))
    if not 0.4 <= duration <= max_seconds:
        raise ValueError(f"Choose a video between 0.4 and {max_seconds} seconds long.")
    if max(video["width"], video["height"]) > 4096:
        raise ValueError("Please export the video at 4K resolution or below.")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-t", str(max_seconds),
               "-vf", "fps=25,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(destination)]
    result = subprocess.run(command, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError("Video conversion failed. Try exporting to H.264 MP4.")
    return duration


def frame_chunks(path, seconds=12):
    cap = cv2.VideoCapture(str(path))
    try:
        frames, start = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) == int(seconds * 25):
                yield start, np.stack(frames)
                start += len(frames)
                frames = []
        if frames:
            yield start, np.stack(frames)
    finally:
        cap.release()


def track_mouth(frames):
    import mediapipe as mp
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models/face_detector.tflite"),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.7)
    landmarks = []
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for index, frame in enumerate(frames):
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), index * 40)
            if len(result.detections) > 1:
                raise ValueError("More than one face is visible. Crop the video to one speaker and retry.")
            if not result.detections:
                landmarks.append(None)
                continue
            face = result.detections[0]
            if min(face.bounding_box.width, face.bounding_box.height) < 60:
                raise ValueError("The face is too small. Use a closer, front-facing recording.")
            h, w = frame.shape[:2]
            landmarks.append(np.array([[p.x * w, p.y * h] for p in face.keypoints[:4]]))
    valid = sum(lm is not None for lm in landmarks)
    gap = longest_gap = 0
    for lm in landmarks:
        gap = gap + 1 if lm is None else 0
        longest_gap = max(gap, longest_gap)
    if valid < len(frames) * 0.8 or longest_gap > 12:
        raise ValueError("The face could not be tracked reliably. Keep one well-lit face visible throughout.")
    crops = VideoProcess(convert_gray=False)(frames, landmarks)
    if crops is None or crops.shape[1:3] != (96, 96):
        raise ValueError("The mouth could not be aligned. Try a front-facing recording.")
    return crops, valid / len(frames)

