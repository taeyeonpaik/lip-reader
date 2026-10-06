# Lip Reader

A Streamlit app that transcribes silent video from mouth movements, with video uploads and webcam recording.

Fine-tuning Auto-AVSR on a custom TCD-TIMIT split reduced test WER from 35.2% to 13.8%.

To get started, run `python tools/bootstrap_dependencies.py`, then follow the [app setup](outputs/lipreading-studio/README.md). You'll need FFmpeg and model weights; weights and datasets aren't included here.
