# LRS3 official-test evaluation

This evaluation uses the retained pretrained Auto-AVSR model and the official LRS3 **test utterances**, recovered from a public processed mirror. It does not create a test set from the local trainval partition. It performs no fine-tuning and uses no LRS2 data.

## Test data and verification

- Mirror: https://huggingface.co/datasets/mattymchen/lrs3-test
- Pinned revision: `9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a`.
- Both Parquet shards downloaded and SHA-256 verified: 677,311,360 bytes total.
- Independent reference: LRS3 test CSV linked in https://github.com/umbertocappellazzo/Llama-AVSR (Google Drive ID `17ob4j1AQZo7NOqr1hP6_z5fZ1ldmGVWA`).
- All **1,321 transcript/frame-count pairs**, with multiplicities, match that test list exactly; **9,890 reference words**.
- **1,315 unique per-clip matches**. Six clips contain repeated short phrases with identical frame counts, leaving three ambiguous ID pairs. Those clips are retained with mirror indices and all possible author IDs. No arbitrary assignment or exclusion changes the denominator.
- The official test ID/session sets have **zero overlap** with the 31,956 verified local LRS3 trainval clips. No duplicate test pixel arrays were found.
- The mirror's Hugging Face split label `train` is a packaging label; identity is established using the independent reference, not the hosting label.

The mirror provides 96×96 grayscale mouth crops. It does not document their complete crop-generation procedure or supply original full-face video byte hashes. Consequently, this is a verified test-utterance evaluation with those crops, not a claim to reproduce every detail of the author's published benchmark pipeline. Audio is not read.

## Fixed model and evaluation

Original checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth`, SHA-256 `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`. Current app decoding: beam 5, CTC weight 0.1, no external LM or length bonus. Center crop 88×88 and original normalization. All parameters remain unchanged.

The protocol and configuration are fixed before inference. Every test clip contributes to corpus WER; failed clips would count as empty predictions. An independent edit-distance implementation recomputes WER/CER, and fresh checkpoint loads reproduce three fixed sample predictions. No settings or weights are selected using this evaluation.

## Run and outputs

From the workspace root:

```sh
PYTHONPATH=outputs/lipreading-studio MPLCONFIGDIR=/private/tmp/lipnet-exact-mpl \
  caffeinate -i /private/tmp/grid-clean-env/bin/python -u -m training.evaluate_lrs3_test
```

Progress, complete predictions, verification, fixed config and final `result.json` are in `training/runs/lrs3-official-test-v1/`. The presence of `result.json` with `state: completed` indicates inference and independent review finished. Workspace logs are in `work/lrs3-official-eval/`.

Download and crop caches remain under `/private/tmp/lrs3-official-test`, outside iCloud. `download_lrs3_test.py` and `verify_lrs3_test.py` document reproducible preparation; the latter expects the existing verified trainval manifest for overlap checks. Only sources, labels, metadata and reports are retained in project artifacts, not raw media.

Earlier 8–11% scores on locally held-out **trainval** data are historical. They do not replace this official-test baseline and may reflect exposure during the original author's pretraining. The benchmark baseline is known after this requested evaluation; a later LRS3-only fine-tuning recipe must use separate validation for selection, never this test result.
