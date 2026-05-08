from pathlib import Path
import hashlib,json,zipfile
root=Path('outputs/lipreading-studio');run=root/'training/runs/lrs3-official-test-v1'
result=json.loads((run/'result.json').read_text());verification=json.loads((run/'dataset-verification.json').read_text());m=result['metrics']
assert result['state']=='completed' and m['clips']==1321 and result['independent_edit_counts_match'] and result['checkpoint_unchanged']
assert len(result['reloaded_prediction_checks'])==3 and all(c.get('same_prediction') for c in result['reloaded_prediction_checks'])
text=f'''# Auto-AVSR — official LRS3 test evaluation

**Completed:** the original pretrained model achieved **{m['wer']:.2%} WER** on the 1,321-sample LRS3 test mirror, using the current app's fixed beam-5 decoder.

| Measure | Result |
|---|---:|
| Test utterances | {m['clips']:,} |
| Reference words | {m['reference_words']:,} |
| Word errors | {m['word_errors']:,} |
| Word error rate | {m['wer']:.2%} |
| Character error rate | {m['cer']:.2%} |
| Exact transcripts | {m['exact_sentences']:,} / {m['clips']:,} |
| Inference failures | {m['failures']} |

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
'''
(run/'REVIEW.md').write_text(text)
Path('outputs/LRS3_TEST_RESULTS.md').write_text(text.replace('in this directory','under `lipreading-studio/training/runs/lrs3-official-test-v1/`').replace('`../../LRS3_OFFICIAL_TEST.md`','`lipreading-studio/training/LRS3_OFFICIAL_TEST.md`'))
p=root/'REVIEW.md';s=p.read_text();heading='## Official LRS3 test baseline — September 19\n';intro=f'''{heading}
The unchanged pretrained Auto-AVSR checkpoint achieved **{m['wer']:.2%} WER** ({m['word_errors']:,} / {m['reference_words']:,} word errors; {m['failures']} inference failures) on all **1,321** verified test-list utterances from the public processed LRS3 test mirror, using the current app's beam-5 decoder. The complete transcript/frame-count multiset matches independent author test labels, with no IDs or recordings shared with our verified trainval data. Six repeated-phrase ID assignments remain ambiguous and are retained without exclusions. Mirror crop provenance is limited; this is not a raw-video byte verification or exact reproduction of the authors' benchmark pipeline. All metrics were independently recomputed and fixed sample predictions reproduced after reload. This replaces local trainval scores as our test baseline; no weights changed. [Full review](training/runs/lrs3-official-test-v1/REVIEW.md).

'''
if heading not in s:p.write_text(s.replace('# Lip Reading Studio — review report\n','# Lip Reading Studio — review report\n\n'+intro,1))
files=[p for p in run.iterdir() if p.is_file() and p.name!='process.lock']
files += [root/'training'/n for n in ['evaluate_lrs3_test.py','download_lrs3_test.py','verify_lrs3_test.py','LRS3_OFFICIAL_TEST.md','inputs.py','metrics.py']]
dest=Path('outputs/lrs3-test-evaluation.zip');members={}
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:
  data=p.read_bytes();name=str(p.relative_to(root));z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
 z.writestr('FILE_MANIFEST.json',json.dumps(members,indent=2))
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
assert not any(n.endswith(('.npz','.parquet','.mp4','.pth')) for n in members)
proof=dict(files=len(members)+1,archive=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),all_member_hashes_and_crc_verified=True,raw_media_included=False,model_weights_included=False)
Path('outputs/LRS3_TEST_PACKAGE_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(metrics=m,package=proof)),flush=True)
