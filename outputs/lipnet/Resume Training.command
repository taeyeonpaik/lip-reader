#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
training_python=../../work/venv/bin/python
if [[ ! -x "$training_python" ]]; then
  print 'This launcher expects the original workspace Python environment in work/venv.'
  print 'See README.md to run full_train.py with another Python installation.'
  exit 1
fi
if [[ -f runs/stabilized-training/training-result.json ]]; then
  "$training_python" finalize_run.py --run runs/stabilized-training --cache ../../work/paper-training-cache
  exit 0
fi
training_resume=()
if [[ -f runs/stabilized-training/last.pt ]]; then
  training_resume=(--resume)
fi
/usr/bin/caffeinate -i "$training_python" full_train.py \
  --data ../../work/tutorial-assets/dataset/data --cache ../../work/paper-training-cache \
  --output runs/stabilized-training --epochs 200 --batch-size 12 --feature-norm --dropout 0.25 "${training_resume[@]}"
"$training_python" finalize_run.py --run runs/stabilized-training --cache ../../work/paper-training-cache
