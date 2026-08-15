"""Fail-closed checks for the separate TCD-TIMIT experiment (no ML imports)."""
import hashlib
import re


def validate(rows, provenance):
    if provenance.get('dataset') != 'TCD-TIMIT':
        raise ValueError('Expected TCD-TIMIT provenance')
    for key in ('source', 'transcript_source', 'preprocessing', 'split_policy'):
        if not provenance.get(key):
            raise ValueError('Missing provenance: ' + key)
    if provenance.get('review_complete') is not True:
        raise ValueError('Acquisition, transcript and preprocessing review incomplete')
    speakers, ids, hashes = {}, set(), {}
    counts = dict(train=0, validation=0, test=0)
    for row in rows:
        split = row['split']
        if split not in counts:
            raise ValueError('Unknown split')
        speaker = row['speaker']
        population = row.get('population')
        if population not in ('volunteer', 'lipspeaker'):
            raise ValueError('Unknown speaker population')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', speaker):
            raise ValueError('Expected a verified source speaker ID')
        if row.get('view') != 'straightcam':
            raise ValueError('This experiment uses frontal recordings only')
        speaker = population + ':' + speaker
        if speaker in speakers and speakers[speaker] != split:
            raise ValueError('Speaker occurs across splits: ' + speaker)
        speakers[speaker] = split
        if row['id'] in ids:
            raise ValueError('Duplicate clip ID')
        ids.add(row['id'])
        if not row['text'].strip() or not row.get('transcript_source'):
            raise ValueError('Missing reference transcript or provenance')
        if not isinstance(row['declared_frames'], int) or row['declared_frames'] <= 0:
            raise ValueError('Invalid frame count')
        for field in ('sha256', 'source_sha256'):
            digest = row[field]
            if not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Invalid ' + field)
            previous = hashes.get((field, digest))
            if previous is not None and previous != split:
                raise ValueError('Identical media occurs across splits')
            hashes[field, digest] = split
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError('All three splits must be nonempty')
    return dict(clips=counts, population_counts={p: {s: sum(r['population'] == p and r['split'] == s for r in rows) for s in counts} for p in ('volunteer', 'lipspeaker')}, speakers={s: sorted(k for k, v in speakers.items() if v == s)
                                       for s in counts}, speaker_overlap=0)


def file_sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
