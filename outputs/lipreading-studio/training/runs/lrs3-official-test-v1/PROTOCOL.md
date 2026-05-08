# Official LRS3 test evaluation — pretrained Auto-AVSR

This evaluation was requested before starting the new LRS3-only fine-tuning run. No optimizer or model update runs here.

## Frozen evaluation settings (written before inference)

- Existing original `models/autoavsr.pth`, SHA-256 `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`.
- Existing app decoder: beam 5, CTC weight 0.1, attention weight 0.9, no external language model, no length bonus. No decoder comparison or tuning against this test.
- Every verified official-test utterance: expected 1,321. Do not substitute local trainval clips or silently skip failures.
- Existing grayscale mouth-crop transform: `[T,96,96]` uint8 → center 88×88 → /255 → normalize mean 0.421, std 0.165. Original visual frontend, projection and encoder; CPU beam search. No audio inputs.
- Corpus word error rate is total word edits / total reference words. Save all hypotheses, references, errors and elapsed times. Independently recompute counts after inference; check sample prediction reloads.
- No parameter changes or checkpoint selection from these results. Preserve the original checkpoint, both LipNet models and current app behavior.

## Provenance and limits

The public `mattymchen/lrs3-test` mirror contains two checksum-verified Parquet shards at revision `9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a`. Its hosting split name `train` is not accepted as a dataset identity. Before inference, all 1,321 transcript/frame-count pairs, including their multiplicities, must match the independent test label file linked by the Llama-AVSR authors. Repeated phrases with identical lengths retain ambiguous candidate IDs and stable mirror indices; no guessed IDs or dropped clips. Check test IDs and recording sessions against our verified LRS3 trainval manifest.

The mirror supplies preprocessed 96×96 grayscale mouth crops, not original full-face videos. Crop production is sparsely documented. The resulting score describes the verified official test utterances with these crops and this decoder; it is not claimed to exactly reproduce the authors' preprocessing or published score. This is distinct from the earlier 8–11% local trainval measurements.

Download/crop caches are under `/private/tmp/lrs3-official-test`, outside iCloud. Reports and scripts are retained without raw test media in the distributable source project.
