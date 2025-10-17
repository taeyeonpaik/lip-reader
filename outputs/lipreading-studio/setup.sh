#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first (on macOS: brew install ffmpeg).'; exit 1; }
command -v ffprobe >/dev/null || { echo 'FFprobe is required alongside FFmpeg.'; exit 1; }
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo 'Setup complete. Open Start Lip Reading.command or run .venv/bin/python -m streamlit run app.py --server.port 8502'
