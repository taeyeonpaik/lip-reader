"""Deterministic recording-group split of LRS3 official-test utterances.

Membership uses source identities only, never predictions or error rates.
Ambiguous identities connect all possible recordings into one indivisible group.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SALT='lrs3-official-holdout-v2-seed20260920'


def partition(test_rows):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    possibilities={}
    for row in test_rows:
        sessions={identity.split('/')[1] for identity in row['possible_author_ids']}
        if row.get('session'): sessions.add(row['session'])
        assert sessions
        possibilities[row['id']]=sessions
        ordered=sorted(sessions)
        for session in ordered: union(ordered[0],session)
    components={}
    for session in parent: components.setdefault(find(session),set()).add(session)
    keys={session:'|'.join(sorted(group)) for group in components.values() for session in group}
    ordered_groups=sorted(set(keys.values()),key=lambda k:hashlib.sha256((SALT+':'+k).encode()).hexdigest())
    validation_groups=set(ordered_groups[:math.ceil(len(ordered_groups)*.2)])
    assigned=[]
    for original in test_rows:
        row=dict(original)
        groups={keys[s] for s in possibilities[row['id']]};assert len(groups)==1
        group=groups.pop()
        row.update(split='validation' if group in validation_groups else 'test',
                   evaluation_group=group,source_hash_format='npz-file',original_partition='official-test')
        assigned.append(row)
    memberships={s:set().union(*(possibilities[r['id']] for r in assigned if r['split']==s)) for s in ('validation','test')}
    assert not memberships['validation'] & memberships['test']
    report=dict(rule='SHA256 ordering of recording components; first ceil(20% of components) validation; rest final test',
                seed_salt=SALT,component_count=len(ordered_groups),validation_components=len(validation_groups),
                counts={s:sum(r['split']==s for r in assigned) for s in ('validation','test')},
                recording_counts={s:len(v) for s,v in memberships.items()},recording_overlap=0,
                ambiguous_rows=sum(r['session'] is None for r in assigned),
                ambiguous_policy='Keep every possible recording for an ambiguous row in the same component; do not guess the true identity',
                predictions_used_for_partition=False)
    return assigned,report


def build():
    old=json.loads((ROOT/'training/runs/lrs3-only-v1/manifest.json').read_text())
    train=[dict(r,source_hash_format='npy-member',original_partition='trainval') for r in old if r['split']=='train']
    original_test=json.loads((ROOT/'training/runs/lrs3-official-test-v1/manifest.json').read_text())
    assert len(train)==30468 and len(original_test)==1321
    evaluation,report=partition(original_test)
    train_sessions={r['session'] for r in train}
    official_sessions={identity.split('/')[1] for r in evaluation for identity in r['possible_author_ids']}
    assert not train_sessions & official_sessions
    report.update(training_clips=len(train),old_trainval_validation_clips_excluded=1488,
                  full_official_test_benchmark=False,
                  prior_exposure='Original beam5 checkpoint evaluated on all 1321 clips before this split; no gradient exposure. Split membership ignores predictions. Model selection now uses only the validation partition.',
                  pretraining_note='Official test membership is held out under the authors published dataset protocol; no independent full audit of every original pretraining source.')
    return train+evaluation,report
