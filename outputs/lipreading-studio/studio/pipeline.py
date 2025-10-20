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


class LipReader:
    def __init__(self, device="auto", beam_size=5, adapter_path=None, lm_path=None,
                 lm_weight=0.0, ctc_weight=0.1, length_bonus=0.0):
        torch.set_num_threads(4)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.beam_size = beam_size
        self.model_origin = "Auto-AVSR author pretrained visual-only"
        self.lock = threading.Lock()
        units = (VENDOR / "spm/unigram/unigram5000_units.txt").read_text().splitlines()
        self.tokens = ["<blank>"] + [line.split()[0] for line in units] + ["<eos>"]
        self.model = E2E(len(self.tokens), "video")
        self.model.load_state_dict(torch.load(ROOT / "models/autoavsr.pth", map_location="cpu", weights_only=True))
        if adapter_path is not None:
            from studio.adaptation import load_adapter
            load_adapter(self.model, adapter_path)
            self.model_origin += " with locally fine-tuned output layers"
        self.model.eval()
        # Encoding benefits from a GPU. Beam search stays on CPU: its many
        # small, sequential operations are inefficient on Apple's GPU.
        for module in (self.model.frontend, self.model.proj_encoder, self.model.encoder):
            module.to(device)
        self.language_model = None
        self.language_model_path = None
        self.configure_decoding(beam_size, lm_path, lm_weight, ctc_weight, length_bonus)

    def configure_decoding(self, beam_size=5, lm_path=None, lm_weight=0., ctc_weight=.1, length_bonus=0.):
        if not isinstance(beam_size, int) or not 1 <= beam_size <= 80:
            raise ValueError("Beam width must be an integer from 1 to 80")
        if not all(np.isfinite(v) for v in [lm_weight, ctc_weight, length_bonus]) or not 0 <= ctc_weight <= 1 or not 0 <= lm_weight <= 2:
            raise ValueError("Invalid decoding weights")
        if lm_weight and lm_path is None:
            raise ValueError("A language model is required when its weight is positive")
        with self.lock:
            if lm_weight and str(lm_path) != self.language_model_path:
                from studio.language_model import load_language_model
                self.language_model = load_language_model(lm_path, self.tokens)
                self.language_model_path = str(lm_path)
            scorers = self.model.scorers()
            scorers["length_bonus"] = LengthBonus(len(self.tokens))
            weights = {"decoder": 1.-ctc_weight, "ctc": ctc_weight, "length_bonus": length_bonus}
            if lm_weight:
                scorers["lm"] = self.language_model
                weights["lm"] = lm_weight
            self.search = BatchBeamSearch(beam_size=beam_size, vocab_size=len(self.tokens),
                weights=weights, scorers=scorers, sos=len(self.tokens)-1,
                eos=len(self.tokens)-1, token_list=self.tokens,
                pre_beam_score_key=None if ctc_weight == 1. else "decoder").eval()
            self.beam_size = beam_size
            self.decoding = {"beam_size": beam_size, "lm_weight": lm_weight,
                             "ctc_weight": ctc_weight, "length_bonus": length_bonus}

    def encode(self, crops):
        """Return frozen visual features; decoding never receives the audio track."""
        if crops.ndim != 4 or crops.shape[1:] != (96, 96, 3) or len(crops) == 0:
            raise ValueError("Expected nonempty 96×96 RGB mouth crops.")
        # Match the author's RGB grayscale transform, center crop and training statistics.
        pixels = torch.from_numpy(crops[:, 4:92, 4:92].copy()).float() / 255
        pixels = (pixels[..., 0] * 0.2989 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114)
        pixels = ((pixels - 0.421) / 0.165).unsqueeze(0).unsqueeze(2).to(self.device)
        with self.lock, torch.inference_mode():
            encoded = self.model.proj_encoder(self.model.frontend(pixels))
            encoded, _ = self.model.encoder(encoded, None)
            return encoded.squeeze(0).cpu()

    def decode(self, encoded):
        with self.lock, torch.inference_mode():
            hypotheses = self.search(encoded.cpu())
        if not hypotheses:
            return ""
        ids = hypotheses[0].yseq.tolist()[1:]
        return "".join(self.tokens[i] for i in ids if 0 < i < len(self.tokens)-1).replace("▁", " ").strip()

    def recognize(self, crops):
        return self.decode(self.encode(crops))

    def transcribe(self, normalized_video, on_segment=None):
        start_time = time.perf_counter()
        segments = []
        for start_frame, frames in frame_chunks(normalized_video):
            crops, coverage = track_mouth(frames)
            text = self.recognize(crops)
            segment = {"start": start_frame / 25, "end": (start_frame + len(frames)) / 25,
                       "text": text, "face_coverage": coverage}
            segments.append(segment)
            if on_segment:
                on_segment(segment, crops[len(crops)//2])
        if not segments:
            raise ValueError("No video frames could be read.")
        return {"text": " ".join(s["text"] for s in segments), "segments": segments,
                "processing_seconds": time.perf_counter() - start_time,
                "video_seconds": segments[-1]["end"], "device": self.device,
                "model": f"{self.model_origin}; beam width {self.beam_size}",
                "decoding": dict(self.decoding),
                "audio_used": False}


