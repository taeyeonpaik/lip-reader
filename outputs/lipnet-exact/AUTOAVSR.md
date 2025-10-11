# Auto-AVSR alongside LipNet

The local app at http://127.0.0.1:8502 now offers three upload/webcam choices:

- **LipNet · fine-tuned**: reviewed epoch 4; GRID-format sentences; three-second takes.
- **LipNet · original pretrained**: preserved author weights and the same GRID mode.
- **Auto-AVSR · unrestricted English**: retained author pretrained weights; natural English; eight-second webcam segments followed by inference.

Select Auto-AVSR, open **Webcam practice**, enable webcam controls, then click **Start** and allow camera access. Face the camera in good light and speak normally. Text appears after each segment; segment boundaries can lose words. Both LipNet choices and the original notebook examples remain available. No model was deleted or retrained for this integration.

The pretrained Auto-AVSR checkpoint is credited in the adjacent `lipreading-studio/models/provenance.json`. It is reused in place, without duplicating its roughly 1 GB weight file. Earlier local fine-tuning candidates did not pass selection; this option does not claim improved or newly trained Auto-AVSR weights. LipNet weights and its verified selection are unchanged.

## Local runtime

The app runs in its existing TensorFlow environment. Auto-AVSR runs in a separate local Python process at `/private/tmp/grid-clean-env/bin/python`, using the saved `../lipreading-studio/studio` pipeline and vendor sources. Model and face-detector SHA-256 checks are mandatory at startup. The worker keeps the model loaded between requests and uses the existing beam-5 decoder. No external inference API or remote compute is used.

Set `AUTOAVSR_PYTHON` to a compatible environment's Python to override the default. If rebuilding the temporary environment, install the versions in `../lipreading-studio/requirements.txt`. The app also needs the adjacent `lipreading-studio` directory with its source and retained model assets. The standalone LipNet ZIP excludes these large optional Auto-AVSR assets; LipNet still works independently when they are absent.

Uploads are normalized to silent 25-fps video in temporary storage, removed after inference. Webcam frames travel only through local memory and process pipes; they are not saved. No audio or displayed practice prompt enters either model. Changing the selected model resets the webcam session and its displayed transcript.

Recorded-input integration checks are in `reports/autoavsr-integration.json`; browser checks are recorded separately. These checks do not constitute a physical-camera accuracy test. The website remains local and undeployed.
