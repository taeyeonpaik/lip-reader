# Validation performed

Tested on macOS Apple Silicon, Python 3.13.7, CPU, using PyTorch 2.14.0,
NumPy 2.5.3, OpenCV 5.0.0 and pytest 9.1.1. CUDA execution was not tested.

## Automated tests: 18 passed

The first 17 tests passed together in 19.84 seconds; the additional command-line
integration test passed in 3.08 seconds. Tests cover:

- CTC loss against exhaustive path enumeration, including repeated characters.
- Prefix beam search against exhaustive path probabilities on six random examples.
- Vocabulary handling, repeat/blank collapse, and corpus-weighted error metrics.
- Full model output shape for 75-frame RGB input, and all convolution block shapes.
- Invariance of a short clip's valid outputs when batched with a longer clip,
  including nonzero padding and nonzero layer biases.
- Finite gradients through every parameter of the full model and a weight update.
- Checkpoint reload preserving predictions exactly.
- Actual MJPEG encode/decode, RGB ordering, cropping and normalization.
- Rejection of incorrect frame rate, crop bounds, and excessive clip length.
- Alignment parsing, speaker split integrity, deterministic split generation,
  duplicate rejection, training statistics and variable-length collation.
- Full-model command-line training, resuming, evaluation and prediction using
  generated video files. Two epochs split across a checkpoint give bit-identical
  model parameters to two uninterrupted CPU epochs in this test.

## Synthetic optimisation

| Run | Parameters | Steps | Initial CTC loss | Final CTC loss |
| --- | ---: | ---: | ---: | ---: |
| Small smoke model | 33,912 | 80 | 18.4722 | 0.1987 |
| Full model | 4,571,388 | 5 | 15.5385 | 1.6141 |

The small model decoded both artificial targets (`ab`, `ba`) correctly. The full
model's five-step run decoded `b` for both, so it demonstrates a decreasing loss
and functioning gradients, not successful sequence learning. The input patterns
are colored synthetic frames with noise, not recordings of a speaker.

Exact smoke outputs are saved in `validation/small-smoke.json` and
`validation/full-smoke.json`. The smoke command deliberately does not produce a
checkpoint that could be mistaken for a trained lip-reader.

## Packaging

Editable installation and the `lipnet --help` console entry point succeeded.
The normal setup command is documented in `README.md`. Dependencies used during
verification are recorded in `requirements-tested.txt`; its exact pins describe
this environment, while `pyproject.toml` specifies the supported version ranges.

## Not established by these tests

No GRID videos were used for training or benchmarking. No real-world accuracy,
published-score reproduction, pretrained weights, landmark tracking, or
language-model-assisted decoding is included. The tests establish software and
mathematical pipeline correctness on the listed cases, not lip-reading quality.
