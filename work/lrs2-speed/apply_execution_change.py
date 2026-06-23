from pathlib import Path
import json
p=Path('outputs/lipreading-studio/training/lrs2_finetune.py')
s=p.read_text()
bench=json.loads(Path('work/lrs2-speed/benchmark.json').read_text())
valid=[r for r in bench['reports'] if r.get('finite_gradients')]
confirmation=json.loads(Path('work/lrs2-speed/threads-confirmation.json').read_text())
# Full optimizer-update confirmation did not support changing thread count.
best=next(r for r in valid if r['device']=='cpu' and r['threads']==4)
Path('work/lrs2-speed/selected-execution.json').write_text(json.dumps(best,indent=2)+'\n')
threads=best['threads'];device=best['device']
s=s.replace("    def mode(self, training):\n        self.model.train(training)",f"    def mode(self, training):\n        torch.set_num_threads({threads} if training else 4)\n        self.model.decoder.to('{device}' if training else 'cpu')\n        self.model.criterion.to('{device}' if training else 'cpu')\n        self.model.train(training)")
if device=='mps':
 a=s.index('    def loss(self, row, augment=True):');b=s.index('\n    def predict(',a)
 s=s[:a]+'''    def loss(self, row, augment=True):
        x = self.model.proj_encoder(self.model.frontend(pixels(row, augment)))
        x, _ = self.model.encoder(x, None)
        ids = [self.units.get(p, self.units['<unk>']) for p in self.sp.encode(row['text'], out_type=str)]
        if len(ids)+sum(a == b for a,b in zip(ids,ids[1:])) > x.shape[1]:
            raise ValueError('Reference too long for CTC: '+row['id'])
        labels = torch.tensor([ids], dtype=torch.long)
        length = torch.tensor([x.shape[1]])
        ctc, _ = self.model.ctc(x.cpu(), length, labels)
        labels = labels.to('mps')
        inp, out = add_sos_eos(labels,self.model.sos,self.model.eos,self.model.ignore_id)
        pred, _ = self.model.decoder(inp,target_mask(inp,self.model.ignore_id),x,None)
        return .1*ctc + .9*self.model.criterion(pred,out).cpu()
''' +s[b:]
 # Optimizer restoration must see the same device layout as saved training state.
 s=s.replace("        self.optimizer = torch.optim.AdamW", "        self.model.decoder.to('mps'); self.model.criterion.to('mps')\n        self.optimizer = torch.optim.AdamW")
 # Baseline decoding runs on CPU; re-place decoder before optimizer-state restoration.
 s=s.replace("        baseline=self.evaluate('validation','baseline-validation')", "        baseline=self.evaluate('validation','baseline-validation')\n        self.mode(True)")
s=s.replace("        self.config.update(validation_origin=", f"        self.config.update(execution_training_cpu_threads={threads},execution_training_decoder_device='{device}',execution_evaluation_cpu_threads=4,\n                           decoded_verification_proof_sha256=sha(RUN/'decoded-verification-proof.json'))\n        self.config.update(validation_origin=")
s=s.replace("def seed(value):", "def verify_source(row):\n    assert sha(row['path'])==row['sha256'], 'Source video changed: '+row['id']\n    return row['sha256']\n\n\ndef seed(value):")
s=s.replace("        status('verify-data',completed=0,total=len(self.rows),workers=8)", """        proof=json.loads((RUN/'decoded-verification-proof.json').read_text())
        assert proof['manifest_sha256']==sha(RUN/'manifest.json')
        assert proof['inputs_sha256']==sha(ROOT/'training/inputs.py')
        assert proof['original_data_check_sha256']==sha(RUN/'data-check-before-speedup.json')
        assert proof['original_config_sha256']==sha(RUN/'config-before-speedup.json')
        assert proof['original_training_source_sha256']==sha(RUN/'training-source-before-speedup.py')
        assert proof['original_data_check']['sha256_verified'] and proof['original_data_check']['clips']==len(self.rows)
        status('verify-source-hashes',completed=0,total=len(self.rows),workers=8)""")
s=s.replace("pool.map(verify_cache,self.rows)","pool.map(verify_source,self.rows)")
s=s.replace("status('verify-data',completed=i", "status('verify-source-hashes',completed=i")
s=s.replace("lrs3_clips=0,recording_overlap=0)","lrs3_clips=0,recording_overlap=0,decoded_checks_reused_from_verified_identical_source_bytes=True,proof_sha256=sha(RUN/'decoded-verification-proof.json'))")
p.write_text(s)
print('Selected execution',best)
