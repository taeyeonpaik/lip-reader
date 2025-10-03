"""The tutorial's video preprocessing, vocabulary, model and CTC operations."""

import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras

VOCABULARY = "abcdefghijklmnopqrstuvwxyz'?!123456789 "
BLANK = len(VOCABULARY) + 1
CLASSES = BLANK + 1
VIDEO_SHAPE = (75, 46, 140, 1)


def encode(text):
    text = " ".join(text.lower().split())
    if not text or set(text) - set(VOCABULARY):
        raise ValueError(f"Unsupported or empty transcript: {text!r}")
    return np.asarray([VOCABULARY.index(c) + 1 for c in text], dtype=np.int64)


def tokens_to_text(tokens):
    return "".join(VOCABULARY[int(t) - 1] for t in tokens if 1 <= int(t) <= len(VOCABULARY))


def alignment_text(path):
    words = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"Invalid alignment line: {line!r}")
        start, end = map(int, parts[:2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid alignment timestamps: {line!r}")
        if parts[2] not in {"sil", "sp"}:
            words.append(parts[2])
    text = " ".join(words).lower()
    encode(text)
    return text


def read_video(path):
    """Reproduce the published checkpoint's input arithmetic, including quirks.

    OpenCV BGR bytes are passed to TensorFlow's RGB grayscale operation, as in
    the video. Centering also happens in uint8 before conversion to float.
    Changing either behavior would change the checkpoint's input distribution.
    Returns normalized model input and uint8 mouth images for display.
    """
    cap = cv2.VideoCapture(str(path))
    crops = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if abs(fps - 25) > 0.5:
            raise ValueError(f"Expected 25 fps GRID video, got {fps}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] < 236 or frame.shape[1] < 220:
                raise ValueError("Tutorial crop requires source frames at least 220×236")
            crop = tf.image.rgb_to_grayscale(frame)[190:236, 80:220]
            crops.append(crop)
            if len(crops) > 75:
                raise ValueError("Expected exactly 75 frames; segment the video first")
    finally:
        cap.release()
    if not crops:
        raise ValueError(f"No decodable frames in {path}")
    pixels = tf.stack(crops)
    std = tf.math.reduce_std(tf.cast(pixels, tf.float32))
    if float(std) < 1e-6:
        raise ValueError("Mouth crop has no visual variation")
    normalized = tf.cast(pixels - tf.reduce_mean(pixels), tf.float32) / std
    # Some GRID clips decode to 74 frames. The tutorial's padded_batch fills
    # their missing tail with zeros after normalization; match that behavior.
    normalized = tf.pad(normalized, [[0, 75 - len(crops)], [0, 0], [0, 0], [0, 0]])
    return normalized.numpy(), pixels.numpy()


def build_model():
    """Checkpoint-compatible 3D CNN + two Bi-LSTMs, not the paper's Bi-GRUs."""
    layers = keras.layers
    model = keras.Sequential()
    for index, width in enumerate((128, 256, 75)):
        kwargs = {"input_shape": VIDEO_SHAPE} if index == 0 else {}
        model.add(layers.Conv3D(width, 3, padding="same", **kwargs))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPool3D((1, 2, 2)))
    model.add(layers.TimeDistributed(layers.Flatten()))
    for _ in range(2):
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                                   kernel_initializer="orthogonal")))
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(CLASSES, activation="softmax", kernel_initializer="he_normal"))
    return model


def load_model(weights):
    model = build_model()
    status = model.load_weights(str(weights))
    if status is not None:
        status.assert_existing_objects_matched()
        # The author's checkpoint may also contain optimizer slots.
        status.expect_partial()
    return model


def decode(probabilities, beam_width=1):
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    probabilities = tf.convert_to_tensor(probabilities)
    lengths = tf.fill([tf.shape(probabilities)[0]], tf.shape(probabilities)[1])
    if beam_width == 1:
        decoded, _ = keras.backend.ctc_decode(probabilities, lengths, greedy=True)
        tokens = decoded[0].numpy()
    else:
        # The legacy Keras backend invokes the v1 decoder with merge_repeated
        # enabled. The v2 decoder preserves repeated letters separated by blank.
        logits = tf.transpose(tf.math.log(probabilities + keras.backend.epsilon()), [1, 0, 2])
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, beam_width=beam_width)
        tokens = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    return [tokens_to_text(row) for row in tokens], tokens


def ctc_loss(labels, probabilities):
    """Use actual label lengths; zero is padding/OOV, never the final CTC blank."""
    labels = tf.cast(labels, tf.int64)
    label_lengths = tf.math.count_nonzero(labels, axis=1, keepdims=True, dtype=tf.int64)
    input_lengths = tf.fill(tf.shape(label_lengths), tf.cast(tf.shape(probabilities)[1], tf.int64))
    return keras.backend.ctc_batch_cost(labels, probabilities, input_lengths, label_lengths)


def tutorial_ctc_loss(labels, probabilities):
    """Exact tutorial objective: includes zero-padding among the target labels.

    Retained for matching the demonstrated training run. Prefer ctc_loss for
    new experiments, since padding is not spoken text.
    """
    batch = tf.shape(labels)[0]
    input_lengths = tf.fill([batch, 1], tf.cast(tf.shape(probabilities)[1], tf.int64))
    label_lengths = tf.fill([batch, 1], tf.cast(tf.shape(labels)[1], tf.int64))
    return keras.backend.ctc_batch_cost(labels, probabilities, input_lengths, label_lengths)


def learning_rate(epoch, current):
    return float(current if epoch < 30 else current * math.exp(-0.1))
