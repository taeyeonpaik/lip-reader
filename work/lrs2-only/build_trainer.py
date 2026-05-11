from pathlib import Path
p=Path('outputs/lipreading-studio/training/lrs3_finetune.py');s=p.read_text()
s=s.replace('verified LRS3 only.', 'verified LRS2 splits only.')
s=s.replace("import numpy as np", "import cv2\nimport numpy as np")
s=s.replace("RUN = ROOT / 'training/runs/lrs3-only-v2'", "RUN = ROOT / 'training/runs/lrs2-full-v1'")
s=s.replace("STORAGE = Path('/private/tmp/autoavsr-lrs3-only-v2')", "STORAGE = Path('/private/tmp/autoavsr-lrs2-full-v1')")
s=s.replace("TEST_RUN = ROOT / 'training/runs/lrs3-official-test-v1'", "SOURCE_MANIFEST = WORKSPACE / 'work/corpora/lrs2/manifest.json'")
a=s.index('def verify_cache(row):');b=s.index('\ndef seed(',a)
s=s[:a]+'''def read_rgb(row):
    cap=cv2.VideoCapture(row['path']); frames=[]
    try:
        assert abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01, 'Wrong frame rate: '+row['id']
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape==(96,96,3), 'Invalid mouth crop: '+row['id']
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    assert len(frames)==row['declared_frames'] and frames, 'Wrong frame count: '+row['id']
    return np.stack(frames)


def verify_cache(row):
    assert sha(row['path'])==row['sha256'], 'Source video changed: '+row['id']
    frames=read_rgb(row)
    return hashlib.sha256(frames.tobytes()).hexdigest()

''' + s[b:]
a=s.index('def prepare():');b=s.index('\ndef pixels(',a)
s=s[:a]+'''def prepare():
    RUN.mkdir(parents=True,exist_ok=True); STORAGE.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE_MANIFEST.read_text())
    rows=[dict(r,dataset='LRS2',format='rgb-mp4',session=r['id'].split('/')[1],
               evaluation_group=r['id'].split('/')[1]) for r in source]
    validate_splits(rows)
    assert {s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}==dict(train=45814,validation=1082,test=1243)
    sessions={s:{r['session'] for r in rows if r['split']==s} for s in ('train','validation','test')}
    assert not sessions['train'] & sessions['validation']
    assert not sessions['train'] & sessions['test']
    assert not sessions['validation'] & sessions['test']
    assert all(Path(r['path']).is_file() for r in rows)
    path=RUN/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==rows, 'Manifest changed'
    else: atomic_json(path,rows)
    report=dict(source_manifest_sha256=sha(SOURCE_MANIFEST),
                source_provenance_sha256=sha(SOURCE_MANIFEST.with_name('provenance.json')),
                splits={s:sum(r['split']==s for r in rows) for s in sessions},
                recording_overlap=0,clip_overlap=0,original_distributor_splits_preserved=True,
                official_counts=dict(train=45839,validation=1082,test=1243),
                official_url='https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html',
                limitations=['Official raw files and lists require BBC credentials; they were not downloaded or bypassed.',
                            'Uses existing public processed crop distribution. Training contains 25 fewer clips than the official count; reason not independently established.',
                            'Validation and test counts match official counts; independent official filelist identity comparison unavailable.',
                            'Pretrained checkpoint already trained on LRS2 training data; this alone does not prove exposure to official validation/test.',
                            'Prior LRS2 test results are already known. No test-based selection in this run.'])
    if (RUN/'dataset-protocol.json').exists(): assert json.loads((RUN/'dataset-protocol.json').read_text())==report
    else: atomic_json(RUN/'dataset-protocol.json',report)
    return rows

''' + s[b:]
s=s.replace("    with np.load(row['path'], allow_pickle=False) as data: frames = data['frames']\n    assert frames.dtype == np.uint8 and frames.shape == (row['declared_frames'], 96, 96)", "    frames = read_rgb(row)")
s=s.replace("    x = torch.from_numpy(frames[:, top:top+88, left:left+88].copy()).float() / 255.", "    rgb = torch.from_numpy(frames[:, top:top+88, left:left+88].copy()).float() / 255.\n    x = rgb[...,0]*.2989 + rgb[...,1]*.587 + rgb[...,2]*.114")
s=s.replace("dataset='LRS3 only; verified 30-hour trainval release plus independently verified cropped test mirror'", "dataset='LRS2 only; all available main training crops and original distributor validation/test partitions'")
s=s.replace("pretraining_history='original checkpoint was pretrained on mixed corpora, including LRS2; new run uses zero LRS2 samples'", "pretraining_history='original checkpoint was pretrained on mixed corpora; this new run uses LRS2 only and starts from the same original author weights, not LRS3-adapted weights'")
a=s.index("        self.config.update(validation_origin=");b=s.index('\n    def mode(',a)
s=s[:a]+'''        self.config.update(validation_origin='Original LRS2 validation partition from pinned processed distribution',
                           source_manifest_sha256=sha(SOURCE_MANIFEST),
                           dataset_protocol_sha256=sha(RUN/'dataset-protocol.json'),
                           changed_from_lrs3='Dataset and data-format handling only; no LRS3 weights or optimizer transferred')
''' +s[b:]
a=s.index('    def migrate(');b=s.index('    def execute(',a)
s=s[:a]+s[b:]
s=s.replace("dict(clips=len(self.rows),sha256_verified=True,lrs2_clips=0,recording_overlap=0)","dict(clips=len(self.rows),sha256_verified=True,lrs2_clips=len(self.rows),lrs3_clips=0,recording_overlap=0)")
s=s.replace("        else:\n            state=self.migrate(baseline)\n", "")
Path('outputs/lipreading-studio/training/lrs2_finetune.py').write_text(s)
