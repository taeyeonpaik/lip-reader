from pathlib import Path
p=Path('outputs/lipreading-studio/app.py')
s=p.read_text()
s=s.replace('def get_reader():\n    return LipReader()', '''def load_reader(adapter_path, adapter_modified):
    return LipReader(adapter_path=adapter_path)


def get_reader():
    selection_file = ROOT / "models/selection.json"
    selected = json.loads(selection_file.read_text()) if selection_file.exists() else {}
    adapter = ROOT / "models/english-adapter.pt"
    if selected.get("selected") == "adapted":
        if not adapter.exists():
            raise RuntimeError("The selected model adapter is missing. Restore it or select the pretrained model.")
        return load_reader(str(adapter), adapter.stat().st_mtime_ns)
    return load_reader(None, 0)''')
s=s.replace('It was trained on varied English speech, independently of our LipNet training.', 'It was trained on varied English speech. Local adaptation and measured results are documented in Model & training.')
s=s[:s.index('with training_tab:')]+'''with training_tab:
    st.subheader("Model & measured results")
    st.write("The visual-only Auto-AVSR model reads natural English sentences. It uses pretrained research weights and is a different architecture from the original LipNet tutorial.")
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
            st.write("**App model:** " + ("Pretrained Auto-AVSR with locally adapted output layers." if selected.get("selected") == "adapted" else "Original pretrained Auto-AVSR."))
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
        st.write("The project began with a LipNet tutorial reproduction. The current product uses Auto-AVSR to support natural English. Earlier LipNet experiments are retained separately in the workspace and are not used by this app.")
    st.link_button("Auto-AVSR source and paper", "https://github.com/mpc001/auto_avsr")
'''
p.write_text(s)
