#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if curl --silent --fail http://127.0.0.1:8502/_stcore/health >/dev/null 2>&1; then
  open http://127.0.0.1:8502
  exit 0
fi
if [[ -x /private/tmp/grid-clean-env/bin/python ]]; then
  grid_python=/private/tmp/grid-clean-env/bin/python
elif [[ -x ../../work/venv/bin/python ]]; then
  grid_python=../../work/venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  grid_python=.venv/bin/python
else
  print 'Create .venv and install requirements-grid.txt first.'
  exit 1
fi
exec "$grid_python" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
