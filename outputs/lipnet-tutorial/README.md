# LipNet — GRID sentence practice

A local lip-reading prototype based primarily on [Nicholas Renotte’s tutorial](https://github.com/nicknochnack/LipNet). It reads six-word GRID sentences using the tutorial’s 3D CNN, two bidirectional LSTMs and CTC. The separate [paper implementation](https://github.com/rizkiarm/LipNet) uses GRUs; this project does not claim they are identical.

## Use the app

Open **Start LipNet.command**, or visit http://127.0.0.1:8502 while the server is running.

- **Webcam practice:** generate a sentence, read it first, then capture one three-second take. Audio is disabled.
- **Uploads:** submit a 1–5-second video containing one sentence and one visible face.
- **GRID examples:** run the original fixed-crop tutorial demonstrations.
- **Training & results:** view the completed local experiment and evaluation.

Sentence structure: **command → color → preposition → letter → number → ending**.
For example: “bin red at S nine again.” The app lists every allowed word and samples all 64,000 combinations. The displayed target is used only for scoring after inference; it never reaches the recognizer. Raw and grammar-constrained outputs are both visible.

## Current model and evaluation

The bounded run completed three epochs in 50.6 minutes, including preprocessing and evaluation. No adapted epoch passed validation, so the author’s supplied 96-epoch checkpoint remains active. The split was 600 training clips from speakers 1 and 3, 150 validation clips from speaker 2, and 200 final-test clips from speaker 4. The ambiguous speaker-5 archive was excluded before training.

**The low-WER goal was not achieved.** With automatic alignment, final-test WER was **81.25% with grammar decoding** and **92.92% raw**, with zero exact sentences out of 200. Validation grammar WER went from 80.00% before adaptation to 81.22% at epoch 3. The prototype is experimental and is not a reliable webcam lip reader. See [REVIEW.md](REVIEW.md).

The tutorial checkpoint’s full training history is not published. We can establish speaker separation for our local adaptation, but cannot certify that validation/test speakers were absent from every stage of pretraining. One validation and one test speaker also provide limited evidence about diverse webcam users. No extremely low WER or webcam accuracy is promised.

Automatic face alignment and grammar decoding are explicit additions to the tutorial. A grammatically valid prediction can still be wrong. We retained the author weights because adaptation did not pass the fixed validation rule. See [the experiment recipe](training/GRID_RETURN.md).

## Setup on another local environment

Python 3.13 and FFmpeg/ffprobe are required:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-grid.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

On this Mac, the launcher prefers the identical pinned environment at `/private/tmp/grid-clean-env` to avoid slow cloud-placeholder reads under Documents. Temporary storage may disappear after reboot; recreate an environment if needed.

## Preserved reproduction and history

The original notebook `LipReading.ipynb`, TensorFlow implementation, `tutorial_demo.py`, author weights, and earlier results remain intact in the workspace. The workspace’s `TUTORIAL_README.md` documents that historical reproduction. These historical files are outside this compact current-prototype ZIP. Auto-AVSR and VideoMAE experiments remain separately preserved under `outputs/lipreading-studio`.

Five focused tests cover grammar generation, CTC repeated characters, word errors, and one-take recording. The browser generator and original-example inference have been checked. Independent checks verified all five prediction reports, 950 filename/transcript pairs, disjoint split hashes, 20 changed weight tensors, and six saved/reloaded predictions. Two recorded clips exercised upload and webcam-worker paths; the physical camera remains untested. Nothing is deployed.
