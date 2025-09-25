import json

import torch

from lipnet.cli import main
from lipnet.data import write_manifest
from test_data import make_video


def test_train_resume_evaluate_predict_and_reproducibility(tmp_path, capsys):
    """Real codec I/O -> full model training -> checkpoint -> inference."""
    train_video, val_video = tmp_path / "train.avi", tmp_path / "val.avi"
    make_video(train_video, frames=6)
    make_video(val_video, frames=5)
    train_manifest, val_manifest = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_manifest([{"video": str(train_video), "text": "ab"}], train_manifest)
    write_manifest([{"video": str(val_video), "text": "ba"}], val_manifest)
    common = ["--threads", "2", "train", "--train", str(train_manifest),
              "--val", str(val_manifest), "--batch-size", "1", "--device", "cpu"]
    resumed = tmp_path / "resumed"
    main(common + ["--output", str(resumed), "--epochs", "1"])
    main(common + ["--output", str(resumed), "--epochs", "2",
                   "--resume", str(resumed / "last.pt")])
    uninterrupted = tmp_path / "uninterrupted"
    main(common + ["--output", str(uninterrupted), "--epochs", "2"])
    restored = torch.load(resumed / "last.pt", weights_only=True)
    expected = torch.load(uninterrupted / "last.pt", weights_only=True)
    assert restored["epoch"] == 2
    for name in expected["model"]:
        torch.testing.assert_close(restored["model"][name], expected["model"][name], rtol=0, atol=0)
    output = tmp_path / "evaluation.json"
    main(["--threads", "2", "evaluate", "--manifest", str(val_manifest),
          "--checkpoint", str(resumed / "last.pt"), "--beam-width", "5", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["sentences"] == 1
    assert result["predictions"][0]["reference"] == "ba"
    main(["--threads", "2", "predict", "--video", str(val_video),
          "--checkpoint", str(resumed / "last.pt"), "--beam-width", "5"])
    assert '"frames": 5' in capsys.readouterr().out
