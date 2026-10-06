# LRS2 latency measurements

## Resume wording

Transcribed approximately 3-second pre-cropped LRS2 test videos in a median 0.80 seconds on an NVIDIA A100, measured across 20 clips using beam-40 decoding.

## Evidence and limitations

- Selected 20 test clips closest to 75 frames before timing analysis; median 2.96 seconds, range 2.84–3.12 seconds.
- GPU figure recomputed from the completed final-test evaluation, not a fresh GPU benchmark. Hardware supported by ../batched-training-launch.json and live training checks. GPU encoder and GPU decoder; 0.8010529635-second median.
- Fresh CPU benchmark: 5.4299720285-second median, 6.5127596857-second p95, Intel Xeon 2.60 GHz with 4 allocated vCPUs and 2 inference threads. Two unmeasured warm-up clips. TCD downloads/verification active; shared resource contention possible.
- Same selected epoch-3 model, strict weight reload and checkpoint SHA verified. No training, decoder tuning or checkpoint selection performed.
- Both include video decoding, crop normalization and visual speech recognition. Neither includes face detection, mouth extraction, uploading, or model initialization. The current app's full-video latency is not measured here.
- Do not describe the GPU figure as end-to-end webcam or upload latency. Original app defaults may use a different decoder/device arrangement.
- Clip-level timing files and selection protocol are saved alongside this document. Model loading was separately measured at 8.95 seconds on CPU.
