"""TCD-TIMIT adapter for the existing CUDA trainer; verified crops required.

Run as python -m training.tcd_finetune [--verify-only]. This module does not
download gated recordings, invent transcripts, or allocate cloud machines.
"""
import argparse
import fcntl
import json
import signal
import time
from pathlib import Path

import torch

from training import lrs2_cuda as base, lrs2_batched as batched, tcd_protocol
from training.data import atomic_json
from training.metrics import summarize

ORIGINAL_VERIFY = base.verify_all


def verify_tcd(rows):
    ORIGINAL_VERIFY(rows)
    path = base.RUN / 'data-check.json'
    report = json.loads(path.read_text())
    report.update(dataset='TCD-TIMIT', tcd_clips=len(rows), lrs2_clips=0, lrs3_clips=0)
    atomic_json(path, report)


def configure():
    base.RUN = base.ROOT / 'training/runs/tcd-timit-v1'
    base.STORAGE = base.WORKSPACE / 'checkpoints/tcd-timit-v1'
    base.SOURCE_MANIFEST = base.WORKSPACE / 'work/corpora/tcd-timit/manifest.json'
    base.verify_all = verify_tcd


def pin(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Pinned experiment input changed: ' + str(path))
    else:
        atomic_json(path, value)


def prepare():
    source = base.SOURCE_MANIFEST
    provenance_path = source.with_name('provenance.json')
    if not source.exists() or not provenance_path.exists():
        raise RuntimeError('TCD-TIMIT acquisition/preprocessing not ready. See training/TCD_FINETUNING.md')
    rows = json.loads(source.read_text())
    provenance = json.loads(provenance_path.read_text())
    report = tcd_protocol.validate(rows, provenance)
    audit = json.loads(source.with_name('independent-data-audit.json').read_text())
    failures = json.loads(source.with_name('evaluation-failures.json').read_text())
    assert audit['passed'] and audit['source_manifest_sha256'] == base.sha(source)
    assert failures == audit['evaluation_failures']
    assert len(failures) == 1 and failures[0]['id'] == 'volunteer/56M/sx67'
    assert failures[0]['split'] == 'test' and failures[0]['text'].strip()
    assert sum(r['split'] == 'test' for r in rows) + len(failures) == 882
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = source.parent / path
        if not path.is_file():
            raise ValueError('Missing prepared crop: ' + str(path))
        row.update(path=str(path.resolve()), dataset='TCD-TIMIT', format='rgb-mp4',
                   session=row['population'] + ':' + row['speaker'],
                   evaluation_group=row['population'] + ':' + row['speaker'])
    base.RUN.mkdir(parents=True, exist_ok=True)
    base.STORAGE.mkdir(parents=True, exist_ok=True)
    report.update(source_manifest_sha256=base.sha(source),
                  source_provenance_sha256=base.sha(provenance_path),
                  protocol_sha256=base.sha(tcd_protocol.__file__),
                  evaluation='speaker-disjoint; 59 volunteers; straightcam only; lipspeakers excluded',
                  official_split_claim=False,
                  limitation='Speaker independence within this dataset does not certify all pretraining exposure.')
    pin(base.RUN / 'manifest.json', rows)
    pin(base.RUN / 'dataset-protocol.json', report)
    pin(base.RUN / 'evaluation-failures.json', failures)
    return rows


class Experiment(batched.Experiment):
    def __init__(self, rows):
        super().__init__(rows)
        self.config.update(
            dataset='TCD-TIMIT only; frontal volunteer recordings; three lipspeakers unavailable',
            validation_origin='Speaker-disjoint TCD-TIMIT partition pinned before evaluation',
            prior_test_exposure='No TCD-TIMIT test evaluation before checkpoint selection',
            pretraining_history='Original author checkpoint; no LRS2/LRS3 project-adapted weights transferred',
            migration='Separate TCD-TIMIT run; original author weights and fresh optimizer',
            changed_from_lrs3='Dataset adapter only; reuse existing batched LRS2 training recipe',
            tcd_source_sha256=base.sha(__file__),
            tcd_protocol_sha256=base.sha(tcd_protocol.__file__))
        failures_path = base.SOURCE_MANIFEST.with_name('evaluation-failures.json')
        self.evaluation_failures = json.loads(failures_path.read_text())
        self.config['evaluation_failures_sha256'] = base.sha(failures_path)
        self.config['evaluation_counts'] = {
            s: len(self.splits[s]) + sum(r['split'] == s for r in self.evaluation_failures)
            for s in self.splits}
        self.config['preprocessing_failure_policy'] = 'Retain failed test clip as empty prediction for baseline and selected model'

    def evaluate(self, split, name):
        """Keep failed preprocessing in both baseline and selected test denominators."""
        self.mode(False)
        expected = self.splits[split] + [r for r in self.evaluation_failures if r['split'] == split]
        path = base.RUN / (name + '.jsonl')
        records = []
        if path.exists():
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    assert i == len(lines) - 1
                    break
                assert record['id'] == expected[i]['id'] and record['reference'] == expected[i]['text']
                records.append(record)
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with path.open('a') as stream:
            for row in expected[len(records):]:
                tick = time.monotonic()
                record = dict(id=row['id'], session=row.get('session'),
                              evaluation_group=row.get('evaluation_group'), reference=row['text'])
                if row.get('preprocessing_error'):
                    record.update(prediction='', error=row['preprocessing_error'])
                else:
                    record['prediction'] = self.predict(row)
                record['seconds'] = time.monotonic() - tick
                records.append(record)
                stream.write(json.dumps(record) + '\n'); stream.flush()
                if len(records) % 50 == 0 or len(records) == len(expected):
                    base.status(name, completed=len(records), total=len(expected))
        result = summarize(records)
        atomic_json(base.RUN / (name + '.json'), result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    configure()
    # Check data readiness before constructing a GPU model or optimizer.
    base.RUN.mkdir(parents=True, exist_ok=True)
    with (base.RUN / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare()
        if args.verify_only:
            base.verify_all(rows)
            return
        if (base.RUN / 'result.json').exists():
            print('Already complete; review reports.')
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no training started')
        signal.signal(signal.SIGTERM, base.request_stop)
        signal.signal(signal.SIGINT, base.request_stop)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        Experiment(rows).execute()


if __name__ == '__main__':
    main()
