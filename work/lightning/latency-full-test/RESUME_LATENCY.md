# Full LRS2 test-set latency

## Recommended resume bullet

Benchmarked visual speech recognition across 1,243 pre-cropped LRS2 test videos, achieving 0.56-second median inference latency on an NVIDIA A100 with beam-40 decoding.

## Complete test set

- All 1,243 clip IDs uniquely matched the test manifest; reference texts matched and all timing values were finite and positive. No clips filtered by speed or duration.
- Clip duration: median 1.44 s, mean 1.64 s, range 0.60–5.80 s (declared decoded frame counts / 25 fps).
- Processing latency: median 0.561 s, mean 0.596 s, p95 0.903 s.
- Total video duration: 2,041.28 s; sum of per-clip processing times: 740.56 s. These totals exclude initialization, verification, logging and other work outside each clip timer; they are not overall job wall time.
- Hardware: NVIDIA A100-SXM4-80GB used for the completed selected-checkpoint final evaluation. Checkpoint epoch 3; CUDA encoding and beam-40 decoding.
- These statistics are recomputed from historical selected-test.jsonl timings, not a fresh GPU benchmark. The first clip is included; no warm-up adjustment or outlier removal.
- Scope includes MP4 reading/decoding, crop normalization and visual model/decoder inference. Excludes face detection, mouth extraction, upload transfer and model loading. Not full-app or physical-webcam latency.
- Results and source hashes are in result.json. This full-set claim supersedes the 20-clip approximately-three-second summary for general test-set reporting.
