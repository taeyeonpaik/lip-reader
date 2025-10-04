# Audit against the requested video and repository

## Conclusion

The 200-epoch local run did **not** train the model from the video. It trained a
separate paper-style GRU implementation. The subsequent LayerNorm experiment also
differs from the video and has been paused at epoch 2. Switching architectures
for faster Mac training was a deviation from the user's instruction.

The matching TensorFlow implementation is in this `lipnet-tutorial` project, but
its local training only completed two epochs. That does not reproduce the
successful 96-epoch checkpoint shown in the video.

| Detail | Video / linked notebook | Failed 200-epoch run |
| --- | --- | --- |
| Input | 75 × 46 × 140 grayscale | RGB, 50 × 100 mouth frames |
| Conv3D widths | 128, 256, 75 | 32, 64, 96 |
| Recurrent layers | Two bidirectional LSTMs, 128 units per direction | Two bidirectional GRUs, 256 per direction |
| Output classes | 41 | 28 |
| Trainable parameters | 8,471,924 | 4,571,388 |
| Adam initial learning rate | 0.0001 | 0.0003 |
| CTC target lengths | 40 padded targets | Actual transcript lengths |
| Training status here | Two complete local epochs | 200 epochs, failed accuracy target |

## Direct verification completed September 17

The original notebook's video loader, model-definition cell and CTC-loss cell
were executed in the compatibility TensorFlow runtime. The two models were
loaded with the same published author weights and given the same two GRID clips.

- Both model definitions contain 8,471,924 parameters.
- Preprocessed inputs: maximum absolute difference 0.
- Output probabilities: maximum absolute difference 0.
- Tutorial CTC loss: maximum absolute difference 0.
- Both sample transcriptions matched their references exactly.

The numeric record is `validation/tutorial-fidelity.json`. This verifies the
matching implementation on those examples; it does not demonstrate successful
local training or independent test accuracy. The author's training split is not
known. The earlier ten-clip demonstration had nine exact sentences with the
published checkpoint and carries the same limitation.

## What the video actually demonstrates

At about 56:12 the author sets a 100-epoch run and explains that the supplied
weights are from epoch 96. At about 57:17 he runs only a couple of demonstration
epochs before loading checkpoints. The notebook likewise defines `epochs=100`
and later loads `models/checkpoint`. Two local epochs are not enough to claim
completion of the successful trained tutorial model.

The matching implementation retains the original crop, grayscale arithmetic,
padded-target CTC objective, optimizer learning rate and schedule. Differences
are explicitly limited to the compatibility runtime, reproducible fixed file
split, input checks, and moving 74-frame tail padding into the loader. An optional
corrected-length loss exists but was not used for the two-epoch reproduction.

The GPU PyTorch adapter preserves the tutorial architecture and has prior
inference parity evidence, but its training objective/optimizer must be audited
before it can replace the TensorFlow training path. It must not be described as
an exact training reproduction based on inference parity alone.

## Current product distinction

Uploads and webcam use the authors' pretrained Auto-AVSR model. They do not use
the tutorial model or the failed local GRU checkpoint. Completing the tutorial
model's training remains outstanding. No website has been deployed.

Sources: [video](https://www.youtube.com/watch?v=uKyojQjbx4c),
[original notebook](https://github.com/nicknochnack/LipNet/blob/main/LipNet.ipynb).
