from pathlib import Path
import collections,hashlib,json
import numpy as np
import pyarrow.parquet as pq
import sys
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
from training.metrics import normalize
meta=Path('work/lrs3-official-eval');data=Path('/private/tmp/lrs3-official-test');dest=Path('outputs/lipreading-studio/training/runs/lrs3-official-test-v1');crops=data/'crops';crops.mkdir(exist_ok=True)
labels=[];lookup=collections.defaultdict(list)
for line in (meta/'author-test-labels.csv').read_text().splitlines():
 dataset,path,frames,tokens,text=line.split(',',4)
 assert dataset=='lrs3' and '/test/' in path
 identity=path.split('/test/')[1].removesuffix('.mp4')
 row=dict(id='lrs3-test/'+identity,session=identity.split('/')[0],frames=int(frames),text=normalize(text));labels.append(row);lookup[(row['text'],row['frames'])].append(row)
print('author rows',len(labels),'ambiguous keys',sum(len(v)>1 for v in lookup.values()),flush=True)
assert len(labels)==1321 and len({r['id'] for r in labels})==1321
api=json.loads((meta/'source-api.json').read_text());sources=json.loads((meta/'source-files.json').read_text());records=[];seen=set();allidx=set();unmatched=[];observed=collections.Counter();ambiguities=[]
for source in sources:
 if not source['path'].endswith('.parquet'):continue
 p=data/Path(source['path']).name
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 assert sha==source['lfs']['oid'] and p.stat().st_size==source['size']
 for batch in pq.ParquetFile(p).iter_batches(batch_size=1,columns=['idx','video','label']):
  idx=batch.column('idx')[0].as_py();text=normalize(batch.column('label')[0].as_py());outer=batch.column('video')[0].values
  assert np.all(outer.value_lengths().to_numpy()==96)
  middle=outer.flatten();assert np.all(middle.value_lengths().to_numpy()==96)
  frames=middle.flatten().to_numpy().reshape(len(outer),96,96)
  assert frames.dtype==np.uint8
  matches=lookup.get((text,len(frames)),[])
  if not matches:
   unmatched.append(dict(index=idx,frames=len(frames),text=text,matches=matches));continue
  observed[(text,len(frames))]+=1
  if len(matches)==1:match=matches[0]
  else:
   ambiguities.append(dict(source_index=idx,text=text,frames=len(frames),possible_ids=[m['id'] for m in matches]))
   match=dict(id='lrs3-test/mirror-index/'+str(idx),session=None)
  assert match['id'] not in seen and idx not in allidx
  seen.add(match['id']);allidx.add(idx)
  out=crops/(str(idx).zfill(4)+'.npz')
  np.savez_compressed(out,frames=frames)
  records.append(dict(id=match['id'],session=match['session'],source_index=idx,split='test',dataset='LRS3-TED',path=str(out),format='gray-npz',declared_frames=len(frames),text=text,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),pixels_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),source_file=source['path'],possible_author_ids=[m['id'] for m in matches]))
  if len(records)%200==0:print('verified',len(records),flush=True)
if unmatched:
 (dest/'unmatched.json').write_text(json.dumps(unmatched,indent=2));raise ValueError(f'{len(unmatched)} records could not be mapped unambiguously')
assert len(records)==1321 and allidx==set(range(1321))
assert observed==collections.Counter((r['text'],r['frames']) for r in labels)
assert {identity for r in records for identity in r['possible_author_ids']}=={r['id'] for r in labels}
trainval=[r for r in json.loads(Path('work/corpora/verified-manifest.json').read_text()) if r['dataset']=='LRS3-TED']
test_stems={r['id'].removeprefix('lrs3-test/') for r in labels};train_stems={r['id'].removeprefix('lrs3/') for r in trainval}
assert not test_stems.intersection(train_stems)
session_overlap={r['session'] for r in labels}&{r['session'] for r in trainval}
assert not session_overlap,session_overlap
pixels=[r['pixels_sha256'] for r in records];assert len(pixels)==len(set(pixels))
records.sort(key=lambda r:r['source_index'])
(dest/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
report=dict(state='verified',clips=len(records),author_test_rows=len(labels),unique_one_to_one_transcript_and_frame_matches=len(records)-len(ambiguities),ambiguous_clip_identity_rows=ambiguities,complete_transcript_frame_multiset_matches=True,test_sessions=len({r['session'] for r in labels}),existing_trainval_clips=len(trainval),clip_overlap_with_local_trainval=0,recording_overlap_with_local_trainval=0,duplicate_pixel_arrays=0,source_repository=api['id'],source_revision=api['sha'],source_hosting_split='train (packaging label only; verified as LRS3 test)',source_files=[r for r in sources if r['path'].endswith('.parquet')],independent_reference='Llama-AVSR author test labels linked from official README',reference_url='https://drive.google.com/file/d/17ob4j1AQZo7NOqr1hP6_z5fZ1ldmGVWA/view',reference_sha256=hashlib.sha256((meta/'author-test-labels.csv').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((dest/'manifest.json').read_bytes()).hexdigest(),input='Mirror-provided 96x96 uint8 grayscale mouth crops, all frames retained; original raw full-face videos not supplied',limitations='Full transcript/frame-count multiset matches the independent official-test list exactly. Six short repeated phrases in three duplicate-key groups have ambiguous per-clip identities; retain mirror indices and all possible author IDs instead of inventing mappings. Original-video byte hashes are unavailable. Mirror crop-generation provenance is sparse; this is evaluation of the official test utterances through these provided crops, not an exact reproduction of author preprocessing. Audio column never loaded. No test result used for tuning.')
(dest/'dataset-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_files'}),flush=True)
