"""Run with: python -m streamlit run app.py"""
import hashlib
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from studio.pipeline import ROOT, LipReader, prepare_video, subtitles, track_mouth
from studio.live import LiveSession

st.set_page_config(page_title="Lip Reading Studio", page_icon="💬", layout="wide")
st.markdown("""<style>
.block-container {max-width:1200px; padding-top:2.6rem}
h1 {letter-spacing:-.04em} [data-testid="stMetricValue"] {font-size:1.65rem}
</style>""", unsafe_allow_html=True)
st.caption("LIP READING STUDIO  /  ON YOUR MAC")
st.title("Turn mouth movements into words.")
st.write("Upload a video or use your webcam. Get an English transcript from mouth movements alone.")


@st.cache_resource
def load_reader(adapter_path, adapter_modified, decoding_json):
    decoding = json.loads(decoding_json)
    return LipReader(adapter_path=adapter_path,
                     lm_path=ROOT / "models/english-lm" if decoding.get("lm_weight", 0) else None,
                     **decoding)


@st.cache_resource
def load_videomae(checkpoint_modified, expected_sha256):
    from studio.videomae import VideoMAEReader, sha256
    checkpoint = ROOT / "models/videomae-ctc.pt"
    if sha256(checkpoint) != expected_sha256:
        raise RuntimeError("The VideoMAE checkpoint failed its integrity check.")
    return VideoMAEReader(checkpoint)


def get_reader():
    if st.session_state.get("reader_choice") == "VideoMAE · experimental":
        selected = json.loads((ROOT / "models/videomae-selection.json").read_text())
        if not selected.get("ready"):
            raise RuntimeError("VideoMAE has not completed verification yet.")
        checkpoint = ROOT / "models/videomae-ctc.pt"
        return load_videomae(checkpoint.stat().st_mtime_ns, selected["checkpoint_sha256"])
    selection_file = ROOT / "models/selection.json"
    selected = json.loads(selection_file.read_text()) if selection_file.exists() else {}
    decoding_file = ROOT / "models/decoding-selection.json"
    decoding = json.loads(decoding_file.read_text())["decoding"] if decoding_file.exists() else {"beam_size": 5, "lm_weight": 0.0}
    decoding_json = json.dumps(decoding, sort_keys=True)
    adapter = ROOT / "models/english-adapter.pt"
    if selected.get("selected") == "adapted":
        if not adapter.exists():
            raise RuntimeError("The selected model adapter is missing. Restore it or select the pretrained model.")
        return load_reader(str(adapter), adapter.stat().st_mtime_ns, decoding_json)
    return load_reader(None, 0, decoding_json)


with st.sidebar:
    st.header("Choose a model")
    videomae_selection = ROOT / "models/videomae-selection.json"
    videomae_ready = (videomae_selection.exists()
                      and json.loads(videomae_selection.read_text()).get("ready")
                      and (ROOT / "models/videomae-ctc.pt").exists())
    choices = ["Auto-AVSR"] + (["VideoMAE · experimental"] if videomae_ready else [])
    if st.session_state.get("reader_choice") not in choices:
        st.session_state["reader_choice"] = choices[0]
    choice = st.selectbox("Transcription model", choices, key="reader_choice")
    if st.session_state.get("previous_reader_choice", choice) != choice:
        previous_live = st.session_state.pop("last_live_session", None)
        if previous_live is not None:
            previous_live.on_ended()
        st.session_state.pop("transcript", None)
        st.session_state.pop("identity", None)
    st.session_state["previous_reader_choice"] = choice
    if choice.startswith("VideoMAE"):
        st.warning("Research-only failed experiment: VideoMAE exceeded 100% word error on evaluation and does not produce usable transcripts. Select Auto-AVSR for the working demo.")
    elif not videomae_ready:
        st.caption("VideoMAE becomes available after training and verification. Progress appears in Model & training; refresh to update this menu.")
    st.divider()
    st.header("A good recording")
    st.write("Face the camera in good light. Keep your whole face visible and speak naturally.")
    st.write("One speaker · English · 0.4–60 seconds · MP4, MOV, WebM or AVI")
    st.caption("Videos are processed locally. Audio is removed before recognition. Uploaded files are deleted from temporary storage after processing.")
    st.divider()
    st.write("Lip reading can miss or invent words. Review the result against what was said.")
    with st.expander("Model details"):
        st.write("The selected model is used for uploads, examples and webcam transcription. Auto-AVSR uses the authors’ lip-reading weights. VideoMAE uses a frozen general-video encoder and a transcription head trained locally from random initialization.")
        st.link_button("Auto-AVSR source", "https://github.com/mpc001/auto_avsr")
        st.link_button("VideoMAE source", "https://github.com/MCG-NJU/VideoMAE")
        st.caption("This is a local research app. Pretrained visual encoders are credited to their authors.")

upload_tab, webcam_tab, examples_tab, training_tab = st.tabs(["Your video", "Live webcam", "Try an example", "Model & training"])
with upload_tab:
    uploaded = st.file_uploader("Upload your video", type=["mp4", "mov", "webm", "avi", "mkv", "mpg"],
                                max_upload_size=100)
    st.caption("Results appear in segments of up to 12 seconds. Processing speed is shown after each run.")
    if uploaded:
        data = uploaded.getvalue()
        identity = hashlib.sha256(data).hexdigest()
        left, right = st.columns([1.15, 1])
        with left:
            st.video(data)
        with right:
            st.subheader(uploaded.name)
            st.write(f"{len(data) / 1024**2:.1f} MB")
            run = st.button("Read this video", type="primary", width="stretch")
        if run:
            wall_start = time.perf_counter()
            st.session_state.pop("transcript", None)
            st.session_state.pop("identity", None)
            try:
                with tempfile.TemporaryDirectory(prefix="lipreading-") as folder:
                    source = Path(folder) / ("upload" + Path(uploaded.name).suffix.lower())
                    source.write_bytes(data)
                    normalized = Path(folder) / "silent.mp4"
                    with st.status("Preparing your video…", expanded=True) as status:
                        duration = prepare_video(source, normalized)
                        st.write("Audio removed. Video prepared at 25 frames per second.")
                        reader = get_reader()
                        st.write(f"Tracking the speaker and reading mouth movements ({reader.device.upper()}).")
                        progress = st.progress(0)
                        def show_segment(segment, crop):
                            progress.progress(min(segment["end"] / duration, 1.0))
                            st.write(f"**{segment['start']:.1f}–{segment['end']:.1f}s** · {segment['text'] or '(no words decoded)'}")
                        result = reader.transcribe(normalized, on_segment=show_segment)
                        result["total_seconds"] = time.perf_counter() - wall_start
                        status.update(label="Transcript ready", state="complete", expanded=False)
                st.session_state["transcript"] = result
                st.session_state["identity"] = identity
            except (ValueError, RuntimeError, OSError) as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Processing failed ({type(error).__name__}). Please try a shorter MP4 recording.")
        if st.session_state.get("identity") == identity:
            result = st.session_state["transcript"]
            st.subheader("Your transcript")
            st.write(result["text"] or "No words were decoded.")
            a, b, c = st.columns(3)
            a.metric("Video length", f"{result['video_seconds']:.1f}s")
            b.metric("Processing time", f"{result['processing_seconds']:.1f}s")
            c.metric("Speed", f"{result['video_seconds']/result['processing_seconds']:.2f}× playback")
            st.caption("Timing covers face tracking and recognition; initial model loading and video conversion are additional.")
            if "total_seconds" in result:
                st.caption(f"Total time after clicking Read: {result['total_seconds']:.1f}s, including conversion and model loading.")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Download transcript", result["text"], "transcript.txt", "text/plain")
            d2.download_button("Download subtitles", subtitles(result), "subtitles.srt", "text/plain")
            d3.download_button("Download details", json.dumps(result, indent=2), "result.json", "application/json")
            st.caption("Subtitle times mark processing segments, not exact word boundaries. Words across a segment boundary may be missed.")

with webcam_tab:
    st.subheader("Read your webcam")
    st.write("Click Start and allow camera access. Face the camera and speak naturally; text appears after each eight-second segment.")
    st.caption("Video only · no microphone · no recordings saved · local processing. There is an eight-second capture window plus processing delay.")
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    live_reader = get_reader()
    def make_live_session():
        def recognize_frames(frames):
            crops, _ = track_mouth(frames)
            return live_reader.recognize(crops)
        return LiveSession(recognize_frames)
    live_context = webrtc_streamer(
        key="lipreading-webcam-" + st.session_state["reader_choice"], mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={"audio": False, "video": {
            "width": {"ideal": 640}, "height": {"ideal": 480},
            "frameRate": {"ideal": 25, "max": 30}, "facingMode": "user"}},
        video_processor_factory=make_live_session, async_processing=True,
        media_toggle_controls=False)
    if live_context.video_processor is not None:
        st.session_state["last_live_session"] = live_context.video_processor

    @st.fragment(run_every=1)
    def live_results():
        session = st.session_state.get("last_live_session")
        if session is None:
            st.info("The camera is off. Start it when you are ready.")
            return
        state = session.snapshot()
        if state["closed"] and not state["busy"] and not state["queued"]:
            st.caption("Camera stopped. Your transcript is below.")
        elif state["busy"]:
            st.caption("Reading the latest segment…")
        else:
            st.caption(f"Collecting a segment: {state['buffered_seconds']:.1f} / 8.0 seconds")
        if state["notice"]:
            st.warning(f"{state['notice']} Skipped segments or interruptions: {state['skipped']}.")
        for segment in state["segments"][-12:]:
            label = f"{segment['start']:.1f}–{segment['end']:.1f}s"
            if segment.get("error"):
                st.warning(f"{label}: {segment['error']}")
            else:
                st.write(f"**{label}** · {segment['text'] or '(no words decoded)'}")
        successful = [s for s in state["segments"] if not s.get("error")]
        if successful:
            result = {"segments": successful}
            st.download_button("Download webcam transcript", "\n".join(s["text"] for s in successful),
                               "webcam-transcript.txt", "text/plain", key="webcam-text")
            st.download_button("Download webcam subtitles", subtitles(result), "webcam-subtitles.srt",
                               "text/plain", key="webcam-srt")
        st.caption("Words that cross segment boundaries may be missed. Up to 300 transcript segments are retained; download before starting another camera session.")
    live_results()

with examples_tab:
    examples = sorted((ROOT / "examples").glob("*.mp4"))
    if examples:
        chosen = st.selectbox("Example", examples, format_func=lambda p: p.stem.replace("-", " ").title())
        st.video(str(chosen))
        if st.button("Read example"):
            with st.spinner("Reading the example…"):
                try:
                    with tempfile.TemporaryDirectory(prefix="lipreading-example-") as folder:
                        normalized = Path(folder)/"silent.mp4"
                        prepare_video(chosen, normalized)
                        result = get_reader().transcribe(normalized)
                    st.write(result["text"])
                    st.caption(f"Processed {result['video_seconds']:.1f}s in {result['processing_seconds']:.1f}s.")
                except Exception as error:
                    st.error(str(error))
    else:
        st.info("Upload a short recording in the Your video tab to try the model.")

with training_tab:
    st.subheader("Model & measured results")
    st.write("Use the sidebar to select a model. Both read video without audio. The experiments below distinguish general-video transfer learning from adaptation of a pretrained lip reader.")

    @st.fragment(run_every=10)
    def videomae_progress():
        run = ROOT / "training/runs/videomae-v6"
        st.subheader("VideoMAE transfer learning · experiment 6")
        st.write("A generic Kinetics-400 video encoder supplies visual features. A new three-layer bidirectional GRU learns character sequences using CTC, which aligns transcripts with video frames during training.")
        st.caption("Frozen encoder · 3.27 million trainable parameters · 5,000 training clips, balanced between LRS2 and LRS3-TED · 400 validation clips · up to 40 epochs within a two-hour training budget, followed by evaluation.")
        result_file = run / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            st.success(f"Training and verification completed: {result['epochs_completed']} epochs; selected epoch {result['selected_epoch']}.")
            if result["heldout"]["macro_wer"] >= 1:
                st.warning("Accuracy outcome: unsuccessful. VideoMAE exceeded 100% word error and recognized no complete heldout sentences correctly. Extra inserted words can push WER above 100%. Engineering checks passed, but the model is unsuitable for useful transcription.")
            records = []
            for split in ["validation", "heldout"]:
                for dataset, measured in result[split]["by_dataset"].items():
                    baseline = result.get("autoavsr_comparison", {}).get(split, {}).get("by_dataset", {}).get(dataset, {})
                    records.append({"Split": split, "Dataset": dataset, "Clips": measured["clips"],
                                    "VideoMAE WER": f"{measured['wer']:.1%}", "VideoMAE CER": f"{measured['cer']:.1%}",
                                    "Auto-AVSR WER, same clips": f"{baseline['wer']:.1%}" if baseline else "Unavailable"})
            st.dataframe(records, hide_index=True, width="stretch")
            st.caption("Lower word error rate (WER) and character error rate (CER) are better. Heldout LRS3 is a local trainval audit, not the official test benchmark. Model selection used validation only. These corpora were used in earlier project experiments.")
            st.download_button("Download VideoMAE results", result_file.read_text(), "videomae-results.json", "application/json")
        elif (run / "failure.json").exists():
            st.error("VideoMAE needs attention: " + json.loads((run / "failure.json").read_text())["error"])
        elif (run / "progress.json").exists():
            progress = json.loads((run / "progress.json").read_text())
            st.info("Running locally · " + progress.get("stage", "starting").replace("_", " "))
            if "completed" in progress and "total" in progress:
                st.progress(progress["completed"] / progress["total"], text=f"{progress['completed']:,} / {progress['total']:,} clips")
            if "epoch" in progress:
                st.write(f"Epoch {progress['epoch']}" + (f" · batch {progress['batch']}/{progress['batches']}" if "batch" in progress else ""))
            if (run / "history.json").exists():
                history = json.loads((run / "history.json").read_text())
                st.dataframe([{"Epoch": h["epoch"], "Training loss": round(h["training_loss"], 3), "Validation WER": f"{h['validation']['macro_wer']:.1%}", "Validation CER": f"{h['validation']['macro_cer']:.1%}"} for h in history], hide_index=True, width="stretch")
        else:
            st.info("Preparing the VideoMAE experiment. Auto-AVSR is available now.")
    videomae_progress()
    st.divider()

    @st.fragment(run_every=10)
    def mwer_progress():
        run = ROOT / "training/runs/mwer-v5"
        if not (run / "config.json").exists():
            return
        st.subheader("Word-error training pilot · experiment 5")
        st.write("1,000 training clips · 400 selection clips · matched conventional-training and MWER arms · two epochs each.")
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            st.success("The bounded pilot has completed.")
            if not result.get("training_started"):
                st.info("The candidate lists did not show enough improvement potential to start the pilot.")
            elif result["selection"]["selected"] == "pretrained":
                st.info("Neither training method met the preset improvement requirements. Auto-AVSR retains its pretrained weights.")
                st.caption("Both methods completed two epochs and 250 updates each. No candidate advanced to the conditional audit/test step. The MWER experiment is finished; VideoMAE progress is reported separately above.")
            verification = ROOT / "validation/mwer-checkpoint-and-metrics.json"
            if verification.exists():
                audit = json.loads(verification.read_text())
                if audit.get("state") == "passed":
                    st.caption("Independent metric and checkpoint checks passed. Both methods show small validation gains, but MWER has no demonstrated advantage over the control. Neither met the preset 0.50-point requirement; test improvement was not established.")
            st.download_button("Download MWER pilot results", (run / "result.json").read_text(), "mwer-pilot-results.json", "application/json")
        elif (run / "progress.json").exists():
            progress = json.loads((run / "progress.json").read_text())
            if progress.get("state") == "failed":
                st.error("Pilot interrupted: " + progress.get("error", "Unknown error"))
            else:
                st.info("Running locally. The active app model is unchanged.")
                st.write("Stage: " + progress.get("stage", "starting").replace("-", " "))
                if "arm" in progress:
                    st.caption(f"Method: {progress['arm']} · epoch {progress['epoch']} / 2")
                if progress.get("total"):
                    st.progress(min(progress["completed"] / progress["total"], 1.))
                    st.caption(f"{progress['completed']:,} / {progress['total']:,} clips")
        if (run / "diagnostic.json").exists():
            diagnosis = json.loads((run / "diagnostic.json").read_text())
            a, b = st.columns(2)
            a.metric("Actual baseline mean WER", f"{diagnosis['baseline']['macro_wer']:.2%}")
            b.metric("Best-candidate mean WER with answers known", f"{diagnosis['oracle']['macro_wer']:.2%}")
            st.caption("The best-candidate score uses the reference answer to choose a transcript. It measures room for improvement within the candidate list; it is not usable model accuracy.")
        histories = []
        for arm in ("ce", "mwer"):
            path = run / (arm + "-history.json")
            if path.exists():
                histories.extend(json.loads(path.read_text()))
        if histories:
            st.table([{"Method": r["arm"], "Epoch": r["epoch"], "LRS2 WER": f"{r['validation']['by_dataset']['LRS2']['wer']:.2%}", "LRS3 WER": f"{r['validation']['by_dataset']['LRS3-TED']['wer']:.2%}", "Mean WER": f"{r['validation']['macro_wer']:.2%}"} for r in histories])
        st.divider()

    mwer_progress()

    @st.fragment(run_every=10)
    def mixed_training_progress():
        run = ROOT / "training/runs/lrs2-lrs3-mixed-v4"
        if not (run / "config.json").exists():
            return
        st.subheader("Combined LRS2 + LRS3 fine-tuning · experiment 4")
        st.write("76,282 training clips · 2,570 validation clips · 55.1 million trainable parameters · up to six epochs.")
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            st.success(f"Training and evaluation completed: {result['epochs_completed']} epochs and {result['updates']:,} optimizer updates.")
            selection = result["selection"]
            history = json.loads((run / "history.json").read_text())
            lowest = min(history, key=lambda r: r["validation"]["macro_wer"])
            a, b = st.columns(2)
            a.metric("Baseline mean validation word error", f"{selection['baseline_validation']['macro_wer']:.2%}")
            b.metric("Lowest observed mean validation word error", f"{lowest['validation']['macro_wer']:.2%}")
            st.caption("The mean gives LRS2 and LRS3 equal weight. The lowest observed mean can still fail the per-dataset selection requirements.")
            if selection["selected"] == "pretrained":
                st.info("No epoch met the preset selection requirements. Auto-AVSR retains its original pretrained weights. Training stopped under the early-stopping rule.")
            st.table([{"Dataset": d, "Baseline WER": f"{base['wer']:.2%}", f"Epoch {lowest['epoch']} WER": f"{lowest['validation']['by_dataset'][d]['wer']:.2%}"} for d, base in selection["baseline_validation"]["by_dataset"].items()])
            if result.get("best_test"):
                st.write(f"LRS2 test WER: {result['baseline_test']['wer']:.2%} baseline; {result['best_test']['wer']:.2%} candidate.")
            else:
                st.caption("No adapted checkpoint qualified for test evaluation. The reported 16.14% test WER uses unchanged pretrained weights with beam 20; the Auto-AVSR option uses the faster beam 5 (16.19%).")
            st.download_button("Download combined training results", (run / "result.json").read_text(), "mixed-training-results.json", "application/json")
        elif (run / "progress.json").exists():
            progress = json.loads((run / "progress.json").read_text())
            if progress.get("state") == "failed":
                st.error("Combined training stopped: " + progress.get("error", "Unknown error"))
            else:
                st.info("Combined fine-tuning is running locally. The app retains the original pretrained weights.")
                st.write("**Stage:** " + progress.get("stage", "starting").replace("-", " ").title())
                if "epoch" in progress:
                    st.write(f"Epoch {progress['epoch']} / {progress['epochs']} maximum")
                if progress.get("total"):
                    st.progress(min(progress["completed"] / progress["total"], 1.))
                    st.caption(f"{progress['completed']:,} / {progress['total']:,} clips")
                st.caption("Last update: " + progress["updated_utc"])
        else:
            st.info("Starting baseline evaluation on both datasets.")
        if (run / "history.json").exists():
            history = json.loads((run / "history.json").read_text())
            if history:
                st.line_chart([{"Epoch": r["epoch"], "LRS2 word error": r["validation"]["by_dataset"]["LRS2"]["wer"], "LRS3 word error": r["validation"]["by_dataset"]["LRS3-TED"]["wer"]} for r in history], x="Epoch")
        st.caption("A qualifying checkpoint must improve mean validation WER by at least 0.5 percentage points without increasing WER or CER on either dataset. LRS3 validation is held out of local training; the base checkpoint already trained on LRS3. This is not the official LRS3 test benchmark.")
        st.divider()

    mixed_training_progress()

    @st.fragment(run_every=10)
    def decoding_progress():
        run = ROOT / "training/runs/decoding-v3"
        if not (run / "config.json").exists():
            return
        st.subheader("Decoding comparison · experiment 3")
        st.write("Comparing wider transcript searches and an external English language model. The visual model weights stay fixed in this experiment.")
        active = ROOT / "models/decoding-selection.json"
        if active.exists():
            profile = json.loads(active.read_text())
            st.caption("Auto-AVSR decoding: " + profile.get("description", str(profile["decoding"])))
        else:
            st.caption("Auto-AVSR decoding: original beam width 5, without an external language model.")
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            st.success("Decoding comparison completed.")
            a, b = st.columns(2)
            a.metric("Original test word error", f"{result['baseline_test']['wer']:.2%}")
            b.metric("Validation-selected decoder test word error", f"{result['selected_test']['wer']:.2%}")
            st.caption(f"Average test processing time: {result['baseline_mean_test_seconds']:.2f}s original; {result['selected_mean_test_seconds']:.2f}s selected. Prepared mouth crops; webcam performance is evaluated separately.")
        elif (run / "progress.json").exists():
            progress = json.loads((run / "progress.json").read_text())
            if progress.get("state") == "failed":
                st.error(progress.get("error", "Evaluation stopped"))
            else:
                st.info("Decoding evaluation is running. Candidate settings have not been activated in the app.")
                st.write("**Stage:** " + progress.get("stage", "starting"))
                if progress.get("total"):
                    st.progress(min(progress["completed"] / progress["total"], 1.))
                    st.caption(f"{progress['completed']} / {progress['total']} clips")
        expansion = ROOT / "training/combined-data-provenance.json"
        if not expansion.exists():
            expansion = ROOT.parents[1] / "work/corpora/progress.json"
        if expansion.exists():
            data = json.loads(expansion.read_text())
            if data.get("state") == "completed":
                st.write("**Expanded data ready:** LRS2 main and LRS3-TED trainval mouth crops.")
                st.json(data["datasets"])
            elif data.get("state") == "failed":
                st.warning("Dataset preparation stopped: " + data.get("error", "Unknown error"))
            else:
                st.caption("Dataset preparation: " + data.get("stage", "starting") + f" · {data.get('completed', 0)} / {data.get('total', '?')} files")
        st.caption("LRS3 trainval is approximately 30 hours, not the full LRS3 pretraining corpus. The base model already trained on LRS2 and LRS3.")
        st.divider()

    decoding_progress()

    @st.fragment(run_every=10)
    def deeper_training_progress():
        deeper = ROOT / "training/runs/lrs2-deeper-v2"
        if not (deeper / "config.json").exists():
            return
        st.subheader("Deeper fine-tuning · experiment 2")
        st.write("12,000 training clips · 55.1 million trainable parameters · final two visual-encoder blocks and final two decoder blocks, plus output layers.")
        result_path = deeper / "result.json"
        progress_path = deeper / "progress.json"
        if result_path.exists():
            report = json.loads(result_path.read_text())
            st.success(f"Experiment finished: {report['epochs_completed']} epochs and {report['updates']:,} optimizer updates.")
            selection = report["selection"]
            a, b = st.columns(2)
            a.metric("Baseline validation word error", f"{selection['baseline_validation']['wer']:.2%}")
            b.metric("Best validation word error", f"{selection['best_validation_wer']:.2%}")
            if report.get("best_test"):
                a, b = st.columns(2)
                a.metric("Baseline test word error", f"{report['baseline_test']['wer']:.2%}")
                b.metric("Candidate test word error", f"{report['best_test']['wer']:.2%}")
            if selection["selected"] == "pretrained":
                st.info("The candidate did not meet the preset validation improvement threshold. Auto-AVSR retains its original pretrained weights.")
            st.caption("The candidate is the checkpoint chosen by validation, not the final training epoch. Lower word error is better; test results do not establish accuracy on arbitrary webcam speakers.")
            st.download_button("Download deeper experiment results", result_path.read_text(), "deeper-training-results.json", "application/json")
        elif progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("state") == "failed":
                st.error("Training stopped: " + progress.get("error", "Unknown error"))
            else:
                st.info("Deeper fine-tuning is running on this Mac. The app currently uses the original pretrained model.")
                st.write("**Stage:** " + progress.get("stage", "starting").replace("-", " ").title())
                if "epoch" in progress:
                    st.write(f"Epoch {progress['epoch']} / {progress['epochs']} maximum")
                if progress.get("total"):
                    st.progress(min(progress["completed"] / progress["total"], 1.))
                    st.caption(f"{progress['completed']:,} / {progress['total']:,} clips")
                st.caption("Last update: " + progress["updated_utc"])
        else:
            st.info("Starting baseline evaluation for the deeper experiment.")
        if (deeper / "history.json").exists():
            history = json.loads((deeper / "history.json").read_text())
            if history:
                st.line_chart([{"Epoch": r["epoch"], "Validation word error": r["validation"]["wer"]} for r in history], x="Epoch")
        st.caption("Selection requires at least a 0.5 percentage-point reduction in validation word error with no increase in character error. Results use prepared LRS2 mouth crops; accuracy on new webcam speakers is not established.")

    deeper_training_progress()
    st.divider()
    st.subheader("Completed output-layer experiment · experiment 1")
    run_dir = ROOT / "training/runs/lrs2-heads-v1"

    @st.fragment(run_every=10)
    def english_training_progress():
        result_path = run_dir / "result.json"
        progress_path = run_dir / "progress.json"
        if result_path.exists():
            report = json.loads(result_path.read_text())
            selection_path = ROOT / "models/selection.json"
            selected = json.loads(selection_path.read_text()) if selection_path.exists() else {"selected": "pretrained"}
            st.success(f"Local fine-tuning experiment completed: {report['epochs_completed']} epochs and {report['training_steps']:,} optimizer updates.")
            st.write("**Auto-AVSR selection:** " + ("Pretrained Auto-AVSR with locally adapted output layers." if selected.get("selected") == "adapted" else "Original pretrained Auto-AVSR."))
            st.caption(report["selection"]["reason"] + ".")
            baseline = report["baseline_test"]
            candidate = report["selected_test"]
            a, b, c = st.columns(3)
            a.metric("Pretrained word error", f"{baseline['wer']:.1%}")
            b.metric("Selected model word error", f"{candidate['wer']:.1%}")
            c.metric("Test clips", f"{candidate['clips']:,}")
            st.caption("Word error counts substitutions, missing words and extra words per reference word; lower is better. These results use prepared LRS2 mouth crops. They do not measure the face tracker or establish accuracy on arbitrary webcam recordings.")
            st.download_button("Download experiment results", result_path.read_text(), "training-results.json", "application/json")
        elif progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("state") == "failed":
                st.error("The local experiment stopped: " + progress.get("error", "Unknown error"))
            else:
                st.info("Local fine-tuning experiment is running. The app continues to use the pretrained model.")
                st.write("**Current stage:** " + progress.get("stage", "starting").replace("_", " ").replace("-", " ").title())
                if "epoch" in progress:
                    st.write(f"Epoch {progress['epoch']} / {progress['epochs']}")
                if progress.get("total"):
                    st.progress(min(progress["completed"]/progress["total"], 1.))
                    st.caption(f"{progress['completed']:,} / {progress['total']:,} clips")
                st.caption("Last update: " + progress["updated_utc"])
        else:
            st.info("The pretrained model is ready. Local experiment results will appear here.")
        history_path = run_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history:
                st.line_chart([{"Epoch": r["epoch"], "Validation word error": r["validation"]["wer"]} for r in history], x="Epoch")
        st.write("**Experiment:** 4,000 training clips, 500 validation clips, and 1,243 test clips. Only the CTC and attention output layers are updated; the visual encoder stays fixed. Validation chooses the model before test evaluation.")
        st.caption("The base model already trained on LRS2. These are local adaptation results, not a new unseen-speaker benchmark or a reproduction of the authors’ published score.")

    english_training_progress()
    with st.expander("How the app works"):
        st.write("Video → face tracking → aligned mouth images → visual encoder → English text. Audio is removed for uploads and disabled for webcam capture.")
        st.write("Webcam results arrive after eight-second capture windows plus processing time. If processing falls behind, the app reports skipped segments.")
    with st.expander("Project origins"):
        st.write("The project began with a LipNet tutorial reproduction. The app offers Auto-AVSR for natural English and adds an experimental VideoMAE model after local training and verification. Earlier LipNet experiments are retained separately in the workspace and are not used by this app.")
    st.link_button("Auto-AVSR source and paper", "https://github.com/mpc001/auto_avsr")
