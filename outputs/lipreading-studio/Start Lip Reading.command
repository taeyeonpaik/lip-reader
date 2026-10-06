#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if curl --silent --fail http://127.0.0.1:8502/_stcore/health >/dev/null 2>&1; then
  open http://127.0.0.1:8502
  exit 0
fi
if [[ -x ../../work/venv/bin/python ]]; then
  studio_python=../../work/venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  studio_python=.venv/bin/python
else
  print 'Run ./setup.sh once, then open this launcher again.'
  exit 1
fi
export MPLCONFIGDIR="$(mktemp -d -t lipreading-matplotlib)"
exec "$studio_python" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless false --browser.gatherUsageStats false
