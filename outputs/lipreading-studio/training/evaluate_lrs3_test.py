"""Evaluate the retained app checkpoint on the verified official LRS3 test utterances."""
import fcntl,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from studio.pipeline import ROOT,LipReader
from training.inputs import load_pixels
from training.metrics import summarize,normalize

RUN=ROOT/'training/runs/lrs3-official-test-v1'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
 path=RUN/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def status(stage,**extra):
 save('progress.json',dict(stage=stage,updated_unix=time.time(),**extra));print(stage,json.dumps(extra),flush=True)
def distance(a,b):
 table=np.empty((len(a)+1,len(b)+1),dtype=np.int32);table[:,0]=np.arange(len(a)+1);table[0,:]=np.arange(len(b)+1)
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):table[i,j]=min(table[i-1,j]+1,table[i,j-1]+1,table[i-1,j-1]+(x!=y))
 return int(table[-1,-1])

def predict(reader,row):
 if sha(row['path'])!=row['sha256']:raise ValueError('Input cache checksum mismatch')
 pixels=load_pixels(row,reader.device)
 with torch.inference_mode():
  x=reader.model.proj_encoder(reader.model.frontend(pixels))
  x,_=reader.model.encoder(x,None)
 return reader.decode(x.squeeze(0).cpu())

def main():
 RUN.mkdir(exist_ok=True);lock=(RUN/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=json.loads((RUN/'manifest.json').read_text());verification=json.loads((RUN/'dataset-verification.json').read_text())
 assert verification['state']=='verified' and len(rows)==1321
 assert all(r['dataset']=='LRS3-TED' and r['split']=='test' and r['id'].startswith('lrs3-test/') for r in rows)
 assert verification['clip_overlap_with_local_trainval']==verification['recording_overlap_with_local_trainval']==0
 original='fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb'
 assert sha(ROOT/'models/autoavsr.pth')==original
 config=dict(checkpoint_sha256=original,manifest_sha256=sha(RUN/'manifest.json'),dataset_verification_sha256=sha(RUN/'dataset-verification.json'),beam_size=5,ctc_weight=.1,lm_weight=0.,length_bonus=0.,input_transform_sha256=sha(ROOT/'training/inputs.py'),pipeline_sha256=sha(ROOT/'studio/pipeline.py'),evaluation_script_sha256=sha(__file__),total_clips=1321,train_updates=0,audio_used=False,test_used_for_tuning=False)
 if (RUN/'config.json').exists():assert json.loads((RUN/'config.json').read_text())==config,'Configuration changed on resume'
 else:save('config.json',config)
 records=[];log=RUN/'predictions.jsonl'
 if log.exists():
  for i,line in enumerate(log.read_text().splitlines()):
   try:record=json.loads(line)
   except json.JSONDecodeError:
    assert i==len(log.read_text().splitlines())-1;break
   assert record['id']==rows[i]['id'] and record['reference']==rows[i]['text'];records.append(record)
  log.write_text(''.join(json.dumps(r)+'\n' for r in records))
 status('loading_model',completed=len(records),total=len(rows));reader=LipReader(beam_size=5);start=time.monotonic()
 with log.open('a') as out:
  for row in rows[len(records):]:
   tick=time.monotonic();record=dict(id=row['id'],session=row['session'],reference=row['text'],prediction='')
   try:record['prediction']=predict(reader,row)
   except Exception as error:record['error']=f'{type(error).__name__}: {error}'
   record['seconds']=time.monotonic()-tick;out.write(json.dumps(record)+'\n');out.flush();records.append(record)
   if len(records)%50==0 or len(records)==len(rows):status('evaluating',completed=len(records),total=len(rows),elapsed_seconds_this_session=time.monotonic()-start)
 metrics=summarize(records);save('metrics.json',metrics)
 status('independent_metric_and_reload_verification',completed=len(records),total=len(rows))
 errors=words=char_errors=chars=exact=0
 for record in records:
  ref=normalize(record['reference']);hyp='' if record.get('error') else normalize(record['prediction'])
  errors+=distance(ref.split(),hyp.split());words+=len(ref.split());char_errors+=distance(ref,hyp);chars+=len(ref);exact+=ref==hyp
 assert (errors,words,char_errors,chars,exact)==(metrics['word_errors'],metrics['reference_words'],metrics['character_errors'],metrics['reference_characters'],metrics['exact_sentences'])
 assert sha(ROOT/'models/autoavsr.pth')==original
 device=reader.device;del reader;gc.collect()
 if torch.backends.mps.is_available():torch.mps.empty_cache()
 fresh=LipReader(beam_size=5);checks=[]
 for i in [0,len(rows)//2,len(rows)-1]:
  expected=records[i]
  if expected.get('error'):checks.append(dict(id=rows[i]['id'],skipped_due_to_reported_failure=True));continue
  actual=predict(fresh,rows[i]);assert actual==expected['prediction'];checks.append(dict(id=rows[i]['id'],same_prediction=True))
 report=dict(state='completed',metrics=metrics,independent_edit_counts_match=True,reloaded_prediction_checks=checks,checkpoint_unchanged=True,checkpoint_sha256=original,device=device,decoder=config,source_verification='dataset-verification.json',training_updates=0,audio_used=False,full_official_test_utterance_count_verified=True,raw_official_video_bytes_verified=False,limitations='Official test utterances independently matched by transcript and frame count. Uses mirror-supplied grayscale mouth crops with sparse preprocessing provenance and app beam5, not an exact author benchmark reproduction. No webcam accuracy claim. No test-based model or decoder selection.',completed_unix=time.time())
 save('result.json',report);status('complete',completed=len(rows),total=len(rows),wer=metrics['wer'],failures=metrics['failures']);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
