"""CPU inference-only comparison with the actual author checkpoint; no updates."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'outputs/lipreading-studio'
sys.path.insert(0, str(PROJECT))

import torch
from studio.pipeline import LipReader, VENDOR
from studio.adaptation import BASE_SHA256
from training import batching, batched_model
from training.test_batching import original_loss
import sentencepiece as spm


def main():
    torch.set_num_threads(2)
    weights = PROJECT/'models/autoavsr.pth'
    def sha():
        with weights.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    assert sha() == BASE_SHA256
    reader = LipReader(device='cpu', beam_size=40)
    torch.set_num_threads(2)
    model = reader.model.eval()
    all_rows = json.loads((ROOT/'work/corpora/lrs2/manifest.json').read_text())
    train = [r for r in all_rows if r['split'] == 'train']
    rows = [next(r for r in train if r['declared_frames'] == length) for length in (13, 18, 22, 30)]
    for row in rows:
        row['path'] = str(ROOT/'work/corpora/lrs2/videos'/Path(row['path']).name)
    sp = spm.SentencePieceProcessor(model_file=str(VENDOR/'spm/unigram/unigram5000.model'))
    units = {line.split()[0]: int(line.split()[-1]) for line in (VENDOR/'spm/unigram/unigram5000_units.txt').read_text().splitlines()}
    tokens = [[units.get(p, units['<unk>']) for p in sp.encode(r['text'], out_type=str)] for r in rows]
    dataset = batching.ClipDataset(rows, tokens, 1)
    clips = [dataset[i] for i in range(4)]
    batch = batching.collate(clips)
    errors, single_losses = [], []
    tick = time.monotonic()
    with torch.inference_mode():
        encoded, _ = batched_model.encode(model, batch['inputs'], batch['lengths'])
        value = float(batched_model.loss(model, batch, 'cpu'))
        for index, clip in enumerate(clips):
            x = model.proj_encoder(model.frontend(clip['inputs'][None]))
            expected, _ = model.encoder(x, None)
            actual = encoded[index, :len(clip['inputs'])]
            error = float((actual-expected[0]).abs().max())
            torch.testing.assert_close(actual, expected[0], rtol=3e-4, atol=3e-4)
            single_losses.append(float(original_loss(model, clip)))
            errors.append(error)
    difference = abs(value-sum(single_losses)/4)
    assert difference < 1e-4
    assert sha() == BASE_SHA256
    report = dict(passed=True, device='cpu', gradients_run=False, source_split='train',
                  frames=[r['declared_frames'] for r in rows], ids=[r['id'] for r in rows],
                  encoder_max_absolute_errors=errors, batch_loss=value, single_mean_loss=sum(single_losses)/4,
                  loss_absolute_difference=difference, seconds=time.monotonic()-tick,
                  original_weights_unchanged=True, gpu_speed_not_measured=True,
                  torch=torch.__version__)
    (ROOT/'work/lightning/batched-model-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
