#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install --no-deps wheels/dlib-20.0.1-cp313-cp313-macosx_26_0_universal2.whl
print 'Setup complete. Run Start LipNet.command.'
