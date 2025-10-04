from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tutorial.core import (BLANK, CLASSES, alignment_text, ctc_loss, decode, encode,
                           learning_rate, load_model, read_video, tokens_to_text)
from tutorial.workflow import discover, distance, split_data

ROOT = Path(__file__).resolve().parents[1]


def test_vocabulary_and_alignment(tmp_path):
    assert BLANK == 40 and CLASSES == 41
    assert tokens_to_text([0, *encode("bin red at s nine again"), -1, BLANK]) == "bin red at s nine again"
    path = tmp_path / "a.align"
    path.write_text("0 10 sil\n10 20 bin\n20 30 red\n30 40 sp\n")
    assert alignment_text(path) == "bin red"
    with pytest.raises(ValueError):
        encode("hello🙂")


def test_ctc_handles_padding_and_repeated_letters():
    logits = tf.Variable([[[-2., 3., -1.], [-1., -2., 4.], [-2., 3., -1.], [-1., -2., 3.]]])
    with tf.GradientTape() as tape:
        p = tf.nn.softmax(logits)
        loss = tf.reduce_mean(ctc_loss(tf.constant([[1, 1, 0, 0]], tf.int64), p))
    assert np.isfinite(loss.numpy())
    assert np.isfinite(tape.gradient(loss, logits).numpy()).all()
    # Spaces/blank handling is tested separately using the 41-class vocabulary.
    p = np.full((1, 4, 41), 0.001, np.float32)
    for t, symbol in enumerate([1, 1, BLANK, 1]):
        p[0, t, symbol] = 1
    p /= p.sum(-1, keepdims=True)
    assert decode(p)[0] == ["aa"]
    assert decode(p, 10)[0] == ["aa"]


def test_scheduler_and_distance():
    assert learning_rate(29, 0.0001) == 0.0001
    assert learning_rate(30, 0.0001) == pytest.approx(0.0001 * np.exp(-0.1))
    assert distance("a b c".split(), "a c".split()) == 1


def test_split_is_stable_and_disjoint(tmp_path):
    videos = tmp_path / "s1"
    labels = tmp_path / "alignments" / "s1"
    videos.mkdir(); labels.mkdir(parents=True)
    for i in range(10):
        (videos / f"clip{i}.mpg").touch()
        (labels / f"clip{i}.align").write_text("0 10 a\n")
    train, val = split_data(tmp_path)
    assert len(train) == 9 and len(val) == 1
    assert split_data(tmp_path) == (train, val)
    assert not {r['video'] for r in train} & {r['video'] for r in val}


def test_real_video_and_pretrained_checkpoint():
    weights = ROOT / "models" / "tutorial.weights.h5"
    if not weights.exists():
        pytest.skip("Download the author's checkpoint first")
    path = ROOT / "samples" / "bras9a" / "bras9a.mpg"
    frames, pixels = read_video(path)
    assert frames.shape == (75, 46, 140, 1)
    # Preserve uint8 subtraction used to train the published checkpoint.
    expected = (pixels - np.uint8(pixels.mean())).astype(np.float32) / pixels.astype(np.float32).std()
    np.testing.assert_allclose(frames, expected, rtol=1e-5, atol=1e-5)
    model = load_model(weights)
    assert model.count_params() == 8471924
    assert model.output_shape == (None, 75, 41)
    probabilities = model(frames[None], training=False)
    np.testing.assert_allclose(tf.reduce_sum(probabilities, -1), 1, atol=1e-5)
    assert decode(probabilities)[0] == ["bin red at s nine again"]
