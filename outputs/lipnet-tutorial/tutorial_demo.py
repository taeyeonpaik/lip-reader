"""Local Streamlit companion: python -m streamlit run app.py."""

import json
from pathlib import Path

import streamlit as st

from tutorial.core import load_model
from tutorial.workflow import predict

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Lip reading · TensorFlow demo", page_icon="💬", layout="wide")
st.title("Read the words behind the movement")
st.caption("Lip-reading tutorial · TensorFlow · GRID speaker 1")

with st.sidebar:
    st.header("About this demo")
    st.write("The model sees 75 grayscale mouth frames. It receives no audio or transcript.")
    st.write("Predictions use Nicholas Renotte’s published 96-epoch checkpoint.")
    st.link_button("Open the tutorial", "https://www.youtube.com/watch?v=uKyojQjbx4c")
    progress = ROOT / "runs" / "two-epochs" / "progress.json"
    completed = progress.parent / "training-result.json"
    if completed.exists():
        st.success("Local two-epoch training run completed")
    elif progress.exists():
        try:
            status = json.loads(progress.read_text())
            st.caption(f"Last saved local training progress: epoch {status['epoch']}/2, batch {status['batch']}/450")
        except (ValueError, KeyError):
            pass

videos = sorted((ROOT / "samples").glob("*/*.mpg"))
weights = ROOT / "models" / "tutorial.weights.h5"
if not videos or not weights.exists():
    st.error("Sample videos or trained weights are missing. Follow the setup steps in README.md.")
    st.stop()


@st.cache_resource
def model_for_demo():
    return load_model(weights)


@st.cache_data
def transcribe(path):
    path = Path(path)
    return predict(model_for_demo(), path, path.with_suffix(".align"))


selected = st.selectbox("Choose a sample video", videos, format_func=lambda p: p.stem)
st.caption("These are controlled dataset examples. Recognition errors are shown alongside successful predictions.")
original, mouth = st.columns([1.2, 1])
with original:
    st.subheader("The source video")
    st.video(str(selected.with_suffix(".mp4")))
with mouth:
    st.subheader("The mouth region")
    st.image(str(selected.parent / (selected.stem + "-mouth.gif")), width=420)
    with st.expander("See the normalized model input"):
        st.image(str(selected.parent / (selected.stem + "-input.gif")), width=420)

if st.button("Transcribe video", type="primary"):
    with st.spinner("Reading the mouth movements…"):
        st.session_state["result"] = transcribe(str(selected))
        st.session_state["selected"] = str(selected)

if st.session_state.get("selected") == str(selected):
    result = st.session_state["result"]
    st.subheader("Predicted words")
    st.markdown(f"### {result['prediction'] or '(no words decoded)'}")
    st.write("**Reference transcript:** " + result["reference"])
    if result["exact_match"]:
        st.success("The prediction matches the reference.")
    else:
        st.warning("The prediction differs from the reference.")
    with st.expander("Inspect the decoded character IDs"):
        st.write(result["tokens"])
    st.caption("The reference is used only for comparison after prediction.")
