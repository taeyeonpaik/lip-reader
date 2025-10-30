"""Run with: python -m streamlit run app.py"""

import hashlib

import json

import tempfile

import time

from pathlib import Path

import streamlit as st

from studio.pipeline import ROOT, LipReader, prepare_video, subtitles, track_mouth

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

def get_reader():
    return load_reader(None, 0, json.dumps({"beam_size": 5, "lm_weight": 0.0}))

upload_tab, = st.tabs(["Your video"])

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
