# English language model

Author-released English subword Transformer language model, downloaded from the
[Auto-AVSR multilingual repository's model zoo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#autoavsr-models).

- [Author archive](https://drive.google.com/file/d/1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv/view)
- 53,742,649 parameters; 16 layers; 5,049 token IDs including special symbols.
- Checkpoint and configuration hashes, and verified vocabulary hash: `provenance.json`.
- Architecture/scoring reference: `espnet/nets/pytorch_backend/lm/transformer.py`
  and `transformer/encoder_layer.py` in that repository (Apache 2.0).
- Local inference implementation: `studio/language_model.py`, based on the ESPnet
  equations and existing vendored Apache-2.0 attention and normalization modules.
  Copyright 2019 Shigeki Karita and ESPnet contributors for the reference layers.
- This checkpoint was trained by the authors. We use it for shallow fusion:
  weighted language-model scores supplement visual CTC/attention scores during
  beam search. It receives candidate text tokens, never audio.
- Preserve the authors' model/data terms; see the source repository. The code
  license does not independently establish rights to its training media.
