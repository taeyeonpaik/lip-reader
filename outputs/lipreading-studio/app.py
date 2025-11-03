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

def get_reader():
    return load_reader(None, 0, json.dumps({"beam_size": 5, "lm_weight": 0.0}))

upload_tab, webcam_tab = st.tabs(["Your video", "Live webcam"])
st.session_state["reader_choice"] = "Auto-AVSR"

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
