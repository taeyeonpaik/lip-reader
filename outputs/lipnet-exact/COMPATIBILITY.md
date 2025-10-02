# Compatibility audit

The original sources remain unchanged under `upstream/`. `engines.py` supplies a small runtime adapter for Python 3.13, TensorFlow 2.21 and legacy `tf_keras` 2.21. This is source-faithful execution on a current runtime, not a claim of identical old package binaries or a reproduction of the complete published benchmark.

## Notebook

The original model-definition cell and the original video loader, alignment loader, loss and scheduler function bodies are executed directly from the pinned notebook. This preserves its OpenCV BGR-to-TensorFlow-grayscale behavior and uint8 arithmetic, fixed crop, LSTMs, vocabulary and greedy decoder. The video gets zero padding to 75 frames as in the original `padded_batch` path. Original 40-padded-target CTC loss is retained rather than replacing it with a corrected-length objective.

The published TensorFlow weights are stored as 20 NumPy arrays for portable loading. Every array was compared byte-for-byte to the original project's author-weight HDF5 file. No trained tensor is converted to PyTorch or fine-tuned. `reports/tutorial-weight-export.json` records both hashes.

## Paper repository

The actual `model2.py` class is instantiated; it is the version used by the repository's training and prediction programs. `model.py` is a different older variant and is not silently substituted.

- Old Keras import locations map to `tf_keras`.
- Keras 2.0.2 GRU defaults are explicitly restored: `hard_sigmoid` recurrent activation and `reset_after=False`. A separate NumPy calculation verifies the reset-before GRU math.
- Keras 2.0.2 CTC loss and decoders use `log(probability + 1e-8)`. The paper adapter preserves this literal constant and TF1-compatible CTC operators, including the original beam decoder's repeated-label behavior. Tests execute the historical backend function bodies and compare their results exactly. Notebook backend defaults remain separate.
- Length tensors remain rank one for batch size one; this fixes the old backend's broad squeeze incompatibility without changing multi-example loss calculations.
- Removed SciPy `imread`/`imresize` calls are supplied by ImageIO/Pillow for the original uint8 RGB path. Resizing was compared byte-for-byte with the historical SciPy implementation. The old NumPy aliases used by scikit-video are restored.
- Original dlib face detection, 68-point predictor, mouth center, horizontal padding, first-frame scale, RGB channel order, width/height transpose and `/255` normalization remain in upstream `videos.py`.
- Inference permits variable T, as the original prediction script builds its model from a video's frame count. This avoids rebuilding the same weighted network for each clip length.
- The dictionary and spelling correction are the repository's originals. A practice prompt is never a decoder input.

The bundled dlib build required two macOS SDK header fixes and explicit PNG/JPEG support; see `build-notes/dlib-macos.patch`. Detector and landmark algorithms were not changed.

## Training and app wrappers

The notebook Windows path parsing is adapted for local paths. The original tutorial dataset pipeline, padded labels, Adam rate and schedule remain. The scheduler result is cast to a Python float for current Keras.

The paper training entry point uses the original model, CTC objective, batch size, Adam settings, and original Curriculum augmentation. A Python 3 Sequence replaces the original Python 2 multiprocessing data generator. Process scheduling, callback/reporting behavior, initialization RNG implementation and epoch ordering are not guaranteed to match a historical run. The original source scripts remain available for inspection.

Uploads are bounded by duration, resolution and frame rate before the unchanged paper video pipeline runs. Webcam transport samples frames to 25 fps and bounds frame size; this is an app input adapter. It then calls the original `Video.from_array`. The recorded-worker comparison does not establish physical-camera accuracy. No microphone, reference text or generated practice sentence enters the model.

## Evidence

- `reports/compatibility-checks.json`: old GRU math, SciPy resize, historical CTC parity, both weight reloads, finite original losses and gradients.
- `reports/tutorial-verification.json`: directly instantiated notebook model parity and complete sampled predictions.
- `reports/paper-verification.json`: every loaded weight tensor equals the original HDF5 tensor; complete sampled predictions.
- `reports/runtime-checks.json`: upload and timestamped recorded-webcam inference, including identical output probabilities.
- `reports/independent-audit.json`: separately recomputed word errors and verified upstream source hashes.

These checks establish the listed implementation properties. They do not certify all numeric behavior of TensorFlow 1, successful full training, or performance on arbitrary webcam users.

## Fine-tuning and reproducible spelling ties

The subsequent six-epoch fine-tuning run uses the same model2, original video pipeline and CTC scoring. The explicit training changes (lower learning rate, gradient accumulation and frozen batch normalization) are documented in FINETUNING.md. The notebook and author weight files remain unchanged.

Final independent reload testing found that upstream `Spell.correction` calls `max` on a set using word probability alone. Equal-probability candidates therefore depend on Python's randomized set iteration order. For example, the same raw `lay gren an m one again` could become `lay green at m one again` or `lay green in m one again` across sessions. Raw neural/CTC predictions matched; spelling ties did not.

`stable_spelling.py` makes one explicit compatibility change: sort candidate words alphabetically before choosing the highest original probability. It preserves the original dictionary, candidate generation and scores. The rule is identical for baseline and fine-tuned weights and accepts no references. It was chosen for determinism, not by testing alternative tie rules for accuracy. Epoch four remains the candidate selected before final test; this fix did not reselect an epoch or trigger retraining.

Original reports remain intact. `*-stable.json` reports and `independent-review.json` contain the corrected, reproducible comparisons. The app's upload/webcam model selector uses this stable spelling policy for both available checkpoints. Historical source-restoration results describe the earlier process-specific tie behavior.
