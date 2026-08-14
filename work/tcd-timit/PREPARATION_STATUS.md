# TCD-TIMIT volunteer preparation

User authorized proceeding with all 59 downloaded volunteers and omitting the
three lipspeakers after repeated HTTP 503 errors. Original archives, pretrained
weights and prior LRS2/LRS3/LipNet runs are preserved.

Custom speaker split: 41 train / 9 validation / 9 test, seed 91. Speaker IDs are
pinned before model inference. Not an official benchmark; shared sentence text
is audited separately. No TCD model predictions or optimization yet.

Preparation runs on the existing CPU Studio, using bundled word-level text
references and the existing app RGB mouth-crop pipeline. Per-speaker inventory
and per-clip hash-verified output caches support restarting. Failures block
review approval. No new GPU allocation is authorized by this preparation step;
a current training spending cap has been requested.

Remote progress: work/corpora/tcd-timit/preparation-progress.json
Remote transcript audit: work/corpora/tcd-timit/source-inventory.json
Remote omissions: data/tcd-timit/download-skipped.json

Pilot passed: two source clips at 30000/1001 fps converted to 25 fps, 96x96 RGB;
tracking coverage 100% for both. Original word references checked and montage
visually inspected (mouth centered). FFmpeg 6.1.1 and MediaPipe 0.10.32 installed
in the existing CPU Studio. Full preparation launched with PID 17145 and log
work/lightning/tcd-preparation.log. This is data preparation, not model training.

Full data audit PASSED: 5,781 hashes and full decoded videos checked; authoritative
word references and disjoint speaker partitions verified. One repeatable test
tracking failure (56M/sx67) is retained as an empty prediction for both models.
Final evaluation denominator: 882 test clips. Counts: 4,018 train, 882 validation,
881 usable test plus one failure. Shared prompts: 588 distinct texts across
partitions. Full report: independent-data-audit.json.

The Studio is CPU-only. GPU training has not started; hardware allocation and a
current spending cap are still needed. No new GPU credits spent by this audit.
