#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ -x /private/tmp/lipnet-exact-tf/bin/python ]]; then
  lipnet_python=/private/tmp/lipnet-exact-tf/bin/python
elif [[ -x .venv/bin/python ]]; then
  lipnet_python=.venv/bin/python
else
  print 'Run the local setup in README.md first.'
  exit 1
fi
exec "$lipnet_python" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
