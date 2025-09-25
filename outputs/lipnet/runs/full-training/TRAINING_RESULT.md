# Local LipNet training result

Training completed 200 full epochs from random initialization.
The validation quality target was **NOT MET**.

- Data: 900 GRID s1 training clips, 100 validation clips.
- Best checkpoint: epoch 16.
- Reloaded validation word error: 91.50%.
- Reloaded validation character error: 61.26%.
- Exact validation sentences: 0.00%.
- Summed epoch time: 5.46 hours.
- Reloaded prediction differences: 0 of 100 clips.
- Inference weights: `model.pt`; resumable training state: `last.pt` and `best.pt`.

Validation selected the checkpoint. This is not an independent test score or the
paper's benchmark. The data contains one speaker and scripted GRID sentences;
these results do not demonstrate general English or unseen-speaker accuracy.
The upload/webcam app uses separately credited pretrained Auto-AVSR weights.
