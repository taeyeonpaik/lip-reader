# Video coverage and execution record

Sources checked: the video description, English captions, the author's complete
notebook, and the companion app source. The notebook has six numbered sections.

| Video section | Implemented | Execution evidence |
| --- | --- | --- |
| 0. Install/import dependencies, check GPU | Yes, TensorFlow + legacy Keras environment | Imports and CPU device detection succeeded; no CUDA device available |
| 1. Download data; load video, crop, normalize, encode alignments | Yes | All 1,000 clips decoded; 995 have 75 frames and five have 74; real transcripts encoded |
| 2. TensorFlow pipeline, batching, split, GIF/frame inspection | Yes | 900/100 stable split; batch/label shapes exercised in the executed notebook; preview files generated |
| 3. Construct CNN/Bi-LSTM network and inspect predictions | Yes | 8,471,924 parameters; 75×41 output; random-weight forward pass and checkpoint loading verified |
| 4. CTC, Adam, scheduler, checkpoint/example callbacks, training | Yes | Two full epochs completed: 900 training and 100 validation clips per epoch; weights and optimizer state saved |
| 5. Load published checkpoint, decode batches and individual real videos | Yes | Ten-clip evaluation, four packaged examples, and notebook outputs recorded |

The source notebook requests 100 epochs in `fit`, but the video explicitly stops
after two and uses an author's checkpoint trained for 96 epochs. The local full
run follows the two-epoch demonstration. A 100-epoch command is provided; no claim
is made that 100 epochs were trained locally.

## Companion app

The video mentions a Streamlit app as future work; its repository includes a
companion app. This project provides a working local version with a sample picker,
video playback, mouth/input previews, transcript comparison and decoded IDs.
Browser verification selected `bras9a` and successfully displayed the prediction
`bin red at s nine again` and the matching reference.

## Verified checks

- The full local run completed in 10,916 seconds. Training loss: 83.35 → 70.44;
  validation loss: 69.30 → 65.82. All 100 validation clips were evaluated afterward:
  99.5% word error, 86.19% character error, zero exact sentences. These two-epoch
  weights remain undertrained and are separate from the usable author checkpoint.

- Five TensorFlow tests passed, including checkpoint inference on a real clip.
- A bounded real-data check ran two single-batch epochs, saved checkpoints, and
  successfully resumed for a third single-batch epoch. Its losses and configuration
  are recorded in `validation/bounded-training-resume.json` and the adjacent CSV.
  Those checks are separate from the full two-epoch job.
- The notebook's eight code cells executed successfully. Its training guard was
  left disabled because the full CLI training job was launched separately.
- The published checkpoint loaded into the recreated architecture with strict
  matching of model objects, then was converted to an inference-only HDF5 file.
- Demonstration metrics and every prediction are preserved under `validation/`.
- Default greedy decoding was retained for published-checkpoint comparisons.
- Implementation differences and compatibility fixes are listed in README.md.

## What is not being claimed

The author's 96-epoch model is not locally trained. A short run using `--steps 1`
is not a full epoch. Small sample results are not a benchmark, and the downloaded
checkpoint's original training split is not known. Full local training is marked
complete only after the requested two full epochs return successfully.
