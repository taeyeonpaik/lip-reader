# Sources and assets

- [Nick Nochnack's LipNet notebook](https://github.com/nicknochnack/LipNet/blob/e5377b4025f51a7cc30bc50cb025aeca7663a14e/LipNet.ipynb), revision `e5377b4025f51a7cc30bc50cb025aeca7663a14e`. Original notebook included; published author weights are represented by an exact 20-array export. The notebook's checkpoint download is the source of these weights. They are not this project's newly trained weights.
- [Rizkiarm LipNet](https://github.com/rizkiarm/LipNet/tree/bec592aa1e66378bc7ab33f6110f1c8a431b3b9a), revision `bec592aa1e66378bc7ab33f6110f1c8a431b3b9a`. Its source, MIT license, unseen and overlapped checkpoint files, dictionary, landmark predictor and original demonstration assets are included unchanged. The unseen checkpoint initializes the local fine-tuned model and remains the app’s original-model fallback.
- [LipNet paper](https://arxiv.org/abs/1611.01599) by Assael, Shillingford, Whiteson and de Freitas. This package executes the linked implementations; it does not claim authorship of the architecture or reproduce the complete paper experiment.
- GRID corpus: verified local s3/s4 recordings supply fine-tuning, with s1 validation and s2 final test. Transcripts follow GRID's six-slot filename encoding. Evaluation IDs and predictions are public review artifacts. The full dataset and private/custom recordings are not included in the ZIP. Only the original repository's supplied demos and three small original tutorial examples are bundled.
- Keras 2.0.2 `recurrent.py` and TensorFlow backend source, plus SciPy 0.19 `pilutil.py`, are retained for historical numerical compatibility checks. The corresponding original license notices are included alongside those files.
- dlib 20.0.1 is built from the PyPI source with the documented macOS SDK changes. Its Boost license is included. The 68-point landmark predictor is the original repository asset; its iBUG 300-W training data has separate noncommercial terms noted by dlib. This local educational prototype is not commercially deployed.

`reports/source-manifest.json` hashes the pinned source and repository assets. `reports/tutorial-weight-export.json` verifies the notebook weight export. The ZIP verification manifest separately hashes every distributed file.

Earlier failed adaptation, Auto-AVSR and VideoMAE reports remain in their historical folders. They are not models or training results of this restored implementation.

## Locally trained derivative

`runs/paper-finetune-v1/best.weights.h5` is this project's fine-tuned derivative of the original rizkiarm model2 unseen-speaker checkpoint: six completed epochs, epoch 4 selected before testing, 1,192 usable GRID s3/s4 clips. Original SHA-256: `b4ea5202691223ff8a91141a014f7477df53e945bd5c4fcae8413691db13d181`. Candidate SHA-256: `42fa9a7051f48e9150967431c32a4a89a50f19e4d324a1d68015fcb8b6368a3d`.

Config, original/effective manifests, source video/label hashes, exclusions and all predictions accompany the review without redistributing the full recordings. The app's stable spelling adapter resolves equal-score ties alphabetically for both checkpoints; upstream source files remain unchanged. This contribution is fine-tuning, reproducibility work and the product interface, not authorship of the architecture, dataset or pretrained initialization.
