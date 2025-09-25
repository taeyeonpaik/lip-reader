# Review of the first full training run

The original run completed 200 epochs and 15,000 optimizer updates on the Mac.
Its total measured training/validation epoch time was 5.46 hours.
The quality target was not met. The verified best checkpoint (epoch 16) had
91.5% word error, 61.26% character error and zero exact validation sentences.
The original checkpoints and reports remain in `runs/full-training`.

## Observed failure

All 100 validation predictions from the best checkpoint were the same text:
`set reed it fiee soon`. This is model collapse, not useful lip reading.
The final epoch lowered CTC loss but did not fix recognition.

Decoded cached mouth images were inspected: they contain the speaker's moving
mouth, vary between clips and retain their frame sequence. Four checked clips
had 74–75 frames and nonzero motion. The train/validation split is unchanged.

On four held-out inputs, the best checkpoint's first GRU input gates had mean
absolute preactivation 24.68; 91.4% exceeded magnitude 5. At epoch 200 the mean
was 50.49 and 96.3% exceeded 5. These large values saturate sigmoid/tanh gates
and are consistent with poor use of visual information. This is a diagnostic
hypothesis, not an isolated causal experiment.

## Correction under test

An optional LayerNorm scales each frame's CNN features before the recurrent
layers. When enabled, recurrent input matrices use Xavier initialization.
Dropout is reduced from 0.5 to 0.25 for the revised training recipe.
The original architecture remains the default and old weights still load.
This practical variant differs from the original paper architecture.

Before another full run, a four-clip training-only check must reach character
error below 5% with four different predictions. That checks whether the model
can fit visual examples; it does not measure generalization or product accuracy.
The full trainer now stops if it remains collapsed to at most two validation
predictions with over 80% word error for five epochs after epoch 30.

The upload/webcam app continues to use separately credited pretrained Auto-AVSR.
The unsuccessful locally trained model is not used for those features.

## Preflight result

The four-clip check passed after 475 updates (4.8 minutes): character error
4.12%, four distinct predictions, CTC loss 0.091. This is training-only
memorization evidence, not held-out accuracy. All 20 model tests passed,
including normalization, padding behavior, gradients and checkpoint reload.
The corrected full run starts from fresh random weights in
`runs/stabilized-training`; preflight weights are not reused.
