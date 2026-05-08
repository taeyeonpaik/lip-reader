# Auto-AVSR — official LRS3 test evaluation

**Completed:** the original pretrained model achieved **27.45% WER** on the 1,321-sample LRS3 test mirror, using the current app's fixed beam-5 decoder.

| Measure | Result |
|---|---:|
| Test utterances | 1,321 |
| Reference words | 9,890 |
| Word errors | 2,715 |
| Word error rate | 27.45% |
| Character error rate | 18.36% |
| Exact transcripts | 508 / 1,321 |
| Inference failures | 0 |

## What was evaluated

- Original Auto-AVSR `vsr_trlrs2lrs3vox2avsp_base.pth`; the checkpoint SHA-256 is unchanged.
- Video only, no audio, no locally trained adapter and no optimizer updates.
- Beam 5, CTC weight 0.1, no external language model or length bonus; settings fixed before predictions.
- All test samples retained; no decoder or model selection used their results.
- Independent edit-distance recomputation matched all WER/CER totals. Fresh checkpoint loading reproduced all three prespecified sample predictions.

## Dataset identity

The public [test mirror](https://huggingface.co/datasets/mattymchen/lrs3-test) was downloaded at revision `9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a`; both published shard checksums matched. The mirror's hosting split is called `train`, but this is a packaging label.

All 1,321 transcript/frame-count pairs, with exact multiplicities, match the independent LRS3 test CSV linked by the [Llama-AVSR authors](https://github.com/umbertocappellazzo/Llama-AVSR). There are 1,315 unique per-clip mappings; six short repeated phrases form three ambiguous ID pairs. We retain their mirror indices and candidate author IDs rather than guessing. All six remain in the metrics. Every possible test ID and recording session is disjoint from our 31,956 verified local trainval clips. No duplicate test pixel arrays were found.

## Interpretation

This is the test-split baseline for this project. The earlier 8–11% LRS3 values came from locally held-out **trainval** data and should not be described as official test performance; the pretrained model may already have seen those clips during original training.

The public mirror supplies preprocessed 96×96 grayscale mouth crops, not original full-face video files. Crop-generation provenance is sparse, and original-video byte identity cannot be checked here. The score therefore describes the official test utterances through these provided crops and our fixed decoder. It is not an exact reproduction of the authors' preprocessing/published result, nor a webcam-accuracy measurement.

The test baseline has now been examined at the user's request. Any later fine-tuning must select its checkpoint on separate LRS3 validation data and must not optimize decoder settings against this test. No LRS2 samples entered this evaluation. Both existing models remain unchanged.

Evidence: `result.json`, `metrics.json`, `predictions.jsonl`, `dataset-verification.json`, `manifest.json` and `PROTOCOL.md` in this directory. Reproduction details are in `../../LRS3_OFFICIAL_TEST.md`. Raw download and crop caches are outside iCloud under `/private/tmp/lrs3-official-test`.
