from pathlib import Path
p=Path('outputs/lipreading-studio/training/lrs2_finetune.py')
s=p.read_text()
s=s.replace('The visual\nnetwork runs on MPS and the text decoder/CTC run on CPU, preserving gradients.\nLarge checkpoints live outside Documents.', 'All training modules run on CUDA; beam search remains on CPU.\nCheckpoints live in persistent Studio storage.')
s=s.replace("RUN = ROOT / 'training/runs/lrs2-full-v1'", "RUN = ROOT / 'training/runs/lrs2-cuda-v1'")
s=s.replace("STORAGE = Path('/private/tmp/autoavsr-lrs2-full-v1')", "STORAGE = WORKSPACE / 'checkpoints/lrs2-cuda-v1'\nDEVICE = 'cuda'")
s=s.replace("    if torch.backends.mps.is_available(): torch.mps.manual_seed(value)", "    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)")
s=s.replace("'mps'", "DEVICE")
s=s.replace("self.config.update(execution_training_cpu_threads=4,execution_training_decoder_device='cpu',execution_evaluation_cpu_threads=4,\n                           decoded_verification_proof_sha256=sha(RUN/'decoded-verification-proof.json'))", "self.config.update(execution_training_cpu_threads=4, execution_training_decoder_device='cuda', execution_evaluation_cpu_threads=4,\n                           migration='Restart from original author weights: Mac temporary LRS2 checkpoint unavailable after reboot',\n                           precision='float32; TF32 disabled', runtime_torch=torch.__version__)")
s=s.replace("        self.model.decoder.to('cpu' if training else 'cpu')\n        self.model.criterion.to('cpu' if training else 'cpu')", "        destination = DEVICE if training else 'cpu'\n        for module in (self.model.decoder, self.model.ctc, self.model.criterion):\n            module.to(destination)")
s=s.replace("return x.cpu()  # Autograd crosses the device boundary.", "return x.to(next(self.model.decoder.parameters()).device)")
s=s.replace("torch.tensor([ids], dtype=torch.long)", "torch.tensor([ids], dtype=torch.long, device=x.device)")
s=s.replace("torch.tensor([x.shape[1]])", "torch.tensor([x.shape[1]], device=x.device)")
a=s.index('        # Validate every cache and reference before starting gradients.')
b=s.index("        baseline=self.evaluate",a)
s=s[:a]+"        verify_all(self.rows)\n"+s[b:]
s=s.replace('torch.mps.empty_cache()', 'torch.cuda.empty_cache()')
s=s.replace('mps_driver_gib=torch.mps.driver_allocated_memory()/1024**3', 'cuda_peak_gib=torch.cuda.max_memory_allocated()/1024**3')
s=s.replace("        tick=time.monotonic();experiment.optimizer.zero_grad(set_to_none=True)", "        tick=time.monotonic();experiment.optimizer.zero_grad(set_to_none=True)")
# Portable paths are resolved from unchanged filenames; original manifest is kept as provenance.
s=s.replace("rows=[dict(r,dataset='LRS2'", "rows=[dict(r,dataset='LRS2'")
s=s.replace("    validate_splits(rows)", "    for row in rows:\n        row['path'] = str(SOURCE_MANIFEST.parent / 'videos' / Path(row['path']).name)\n    validate_splits(rows)")
# Environment requires CUDA for optimization; verification/import check can use CPU.
s=s.replace("    RUN.mkdir(parents=True,exist_ok=True)\n    lock=", "    torch.backends.cuda.matmul.allow_tf32 = False\n    torch.backends.cudnn.allow_tf32 = False\n    if not args.verify_only and not args.check_only and not torch.cuda.is_available():\n        raise RuntimeError('CUDA GPU required. Studio is still on CPU; training was not started.')\n    RUN.mkdir(parents=True,exist_ok=True)\n    lock=")
s=s.replace("parser.add_argument('--smoke',action='store_true');args=parser.parse_args()", "parser.add_argument('--smoke',action='store_true'); parser.add_argument('--verify-only',action='store_true'); parser.add_argument('--check-only',action='store_true'); args=parser.parse_args()")
s=s.replace("        if args.smoke: smoke(rows)", "        if args.verify_only: verify_all(rows)\n        elif args.check_only: check_cpu(rows)\n        elif args.smoke: smoke(rows)")
a=s.index('\ndef main():')
s=s[:a]+'''
def verify_all(rows):
    # Repeat full decode and split checks in this environment. Reuse only after
    # every source file still hashes identically to the pinned manifest.
    report_path = RUN/'data-check.json'
    reusable = False
    if report_path.exists():
        old = json.loads(report_path.read_text())
        reusable = (old.get('source_sha256') == sha(__file__) and
                    old.get('manifest_sha256') == sha(RUN/'manifest.json') and
                    old.get('clips') == len(rows) and old.get('decoded_verified') is True)
    hashes = {}
    status('verify-source-hashes' if reusable else 'verify-decoded-data',completed=0,total=len(rows))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,digest) in enumerate(zip(rows,pool.map(verify_source if reusable else verify_cache,rows)),1):
            assert hashes.get(digest,row['split']) == row['split'], 'Duplicate content across splits'
            hashes[digest] = row['split']
            if i%1000 == 0 or i == len(rows):
                status('verify-source-hashes' if reusable else 'verify-decoded-data',completed=i,total=len(rows))
    atomic_json(report_path,dict(clips=len(rows),sha256_verified=True,decoded_verified=True,
                manifest_sha256=sha(RUN/'manifest.json'),source_sha256=sha(__file__),
                recording_overlap=0,lrs2_clips=len(rows),lrs3_clips=0,
                reused_decode_after_identical_source_hashes=reusable))


def check_cpu(rows):
    global DEVICE
    DEVICE = 'cpu'
    e = Experiment(rows)
    e.mode(False)
    row = sorted(e.splits['train'],key=lambda r:r['declared_frames'])[len(e.splits['train'])//2]
    with torch.inference_mode():
        loss = e.loss(row,augment=False)
        assert torch.isfinite(loss)
        prediction = e.predict(row)
    assert e.config['trainable_parameters'] == 250383410
    assert sha(ROOT/'models/autoavsr.pth') == BASE_SHA256
    atomic_json(RUN/'cpu-readiness.json',dict(passed=True,parameters=e.config['trainable_parameters'],
                sample_id=row['id'],split='train',loss=float(loss),prediction=prediction,
                original_weights_unchanged=True,gradients_run=False,cuda_verified=False))
    status('ready-for-gpu',training_started=False)

''' +s[a:]
Path('outputs/lipreading-studio/training/lrs2_cuda.py').write_text(s)
compile(s,'lrs2_cuda.py','exec')
