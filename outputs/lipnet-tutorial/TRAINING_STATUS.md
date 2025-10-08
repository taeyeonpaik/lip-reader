# Local training status

**COMPLETE** — updated 2026-09-16T17:59:00+00:00

Two full epochs completed: 900 training clips and 100 validation clips per epoch. These locally trained weights are separate from the author's 96-epoch demo weights.

Last saved progress: epoch 2/2, batch 450/450.
Elapsed at that point: 180.6 minutes.
Reported running CTC loss: 70.4397.

Run duration: 3.03 hours.

Completed-epoch values:

```json
{
  "loss": [
    83.345458984375,
    70.43968200683594
  ],
  "val_loss": [
    69.3047103881836,
    65.81696319580078
  ],
  "lr": [
    9.999999747378752e-05,
    9.999999747378752e-05
  ]
}
```

The interactive demo continues to use the author's published checkpoint.
This file is updated by a local status-watcher process while training runs.
It does not imply that an assistant is continuously reasoning in the background.
