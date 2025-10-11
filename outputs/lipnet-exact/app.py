"""Local source-faithful LipNet demo and paper-preprocessing webcam wrapper."""
from pathlib import Path
import json,tempfile
import numpy as np
import streamlit as st
from engines import ROOT,UP,Tutorial,Paper
from live import OneTakeSession, LiveSession
from inputs import inspect_upload
from model_selection import SelectedPaper

st.set_page_config(page_title='Lip reading · LipNet & Auto-AVSR',page_icon='💬',layout='wide')
st.title('Lip reading · LipNet & Auto-AVSR')
st.caption('Choose GRID sentences with LipNet or unrestricted English with Auto-AVSR · local, video-only inference')
@st.cache_resource
def tutorial():return Tutorial()
@st.cache_resource
def paper(choice='reviewed'):return SelectedPaper(choice)
@st.cache_resource
def unrestricted():
    from autoavsr_bridge import AutoAVSR
    return AutoAVSR()
def chosen_reader(choice):return unrestricted() if choice=='autoavsr' else paper(choice)
SLOTS=[('Command',['bin','lay','place','set']),('Color',['blue','green','red','white']),('Preposition',['at','by','in','with']),('Letter',list('abcdefghijklmnopqrstuvxyz')),('Number',['zero','one','two','three','four','five','six','seven','eight','nine']),('Ending',['again','now','please','soon'])]
with st.sidebar:
    checkpoint_choice=st.radio('Upload / webcam model',['reviewed','author','autoavsr'],format_func=lambda v:{'reviewed':'LipNet · fine-tuned','author':'LipNet · original pretrained','autoavsr':'Auto-AVSR · unrestricted English'}[v])
    if checkpoint_choice=='autoavsr':
        st.header('Unrestricted English')
        st.write('Speak a short natural English sentence. No GRID format or practice prompt is required. Keep one well-lit face visible and face the camera.')
        st.caption('Auto-AVSR uses the retained author pretrained weights. Earlier fine-tuning did not meet the improvement threshold. Webcam text follows each eight-second segment plus processing time.')
    else:
        st.header('Sentence structure')
        for name,words in SLOTS:st.write('**'+name+':** '+', '.join(words))
        st.caption('Example: bin red at S nine again. Speak normally, facing the camera in good light.')
    st.write('LipNet examples and both LipNet checkpoints remain available. The selected upload/webcam model is shown with each transcript.')
    st.caption('The practice sentence never enters recognition. The paper model was fine-tuned locally from the credited author weights. The original pretrained model remains available.')
    st.link_button('Notebook source','https://github.com/nicknochnack/LipNet/blob/main/LipNet.ipynb')
    st.link_button('Paper repository','https://github.com/rizkiarm/LipNet')
    st.link_button('Auto-AVSR source','https://github.com/mpc001/auto_avsr')

def show(r,reference=None):
    st.subheader(r['text'] or '(No text decoded)')
    if r.get('checkpoint')!='autoavsr':st.write('Raw decoder output: '+(r['raw'] or '(empty)'))
    st.caption(f"{r['seconds']:.2f}s processing · {r['mode']}")
    if reference:st.write('Reference for comparison only: '+reference)

examples,upload,webcam,results=st.tabs(['Original examples','Upload','Webcam practice','Verified results'])
with examples:
    mode=st.radio('Implementation',['Nick’s notebook','Rizkiarm paper repository'],horizontal=True)
    if mode=='Nick’s notebook':
        choices=sorted((ROOT/'samples').glob('*.mpg'));path=st.selectbox('GRID clip',choices,format_func=lambda p:p.stem)
        st.caption('Original 75 × 46 × 140 grayscale input, published 96-epoch checkpoint, greedy CTC. No automatic alignment.')
        if st.button('Read notebook example'):
            with st.spinner('Running the notebook model…'):
                try:r,_=tutorial().predict(path);show(r,(path.with_suffix('.txt')).read_text())
                except Exception as e:st.error(str(e))
    else:
        choice=st.selectbox('Repository example',['Supplied mouth frames: bbaf2n','Supplied full-face video: swwp2s'])
        if st.button('Read paper example'):
            with st.spinner('Running the paper repository model…'):
                try:
                    mouth=choice.startswith('Supplied mouth');path=UP/'evaluation/samples'/('bbaf2n' if mouth else 'id2_vcd_swwp2s.mpg')
                    r,_=paper('author').predict(path,mouth=mouth);show(r,'bin blue at f two now' if mouth else 'set white with p two soon')
                except Exception as e:st.error(str(e))
with upload:
    if checkpoint_choice=='autoavsr':
        st.write('Upload 0.4–60 seconds of natural English with one clearly visible face. Video is normalized to 25 fps and audio is discarded before recognition.')
    else:
        st.write('Upload one GRID-format sentence at 25 fps, lasting 1–5 seconds, with the longest side at most 640 pixels. Ideally use three seconds at 360 × 288. One face, facing forward.')
    file=st.file_uploader('Video',type=['mpg','mp4','mov','webm','avi'],max_upload_size=50)
    if file and st.button('Read upload'):
        with tempfile.TemporaryDirectory(prefix='lipnet-exact-') as folder,st.spinner('Loading the selected model and reading mouth movements…'):
            path=Path(folder)/('video'+Path(file.name).suffix);path.write_bytes(file.getvalue())
            try:
                if checkpoint_choice!='autoavsr':inspect_upload(path)
                model=chosen_reader(checkpoint_choice)
                if getattr(model,'selection_error',None):st.info('Verified fine-tuned weights are unavailable; using the original pretrained model.')
                r,_=model.predict(path);show(r)
            except Exception as e:st.error(str(e))
with webcam:
    import random
    is_auto=checkpoint_choice=='autoavsr'
    if 'prompt' not in st.session_state:st.session_state.prompt=' '.join(random.SystemRandom().choice(w) for _,w in SLOTS)
    if 'take' not in st.session_state:st.session_state.take=0
    if st.session_state.get('checkpoint_choice')!=checkpoint_choice:
        old=st.session_state.pop('worker',None)
        if old:old.on_ended()
        st.session_state.checkpoint_choice=checkpoint_choice;st.session_state.take+=1;st.session_state.pop('reference',None)
    if is_auto:
        st.subheader('Auto-AVSR · unrestricted English webcam')
        st.write('Enable webcam controls, click Start and allow camera access. Speak naturally with one well-lit face visible. A transcript appears after each eight-second segment plus processing time.')
        st.caption('No fixed sentence structure. First use loads the retained model. LipNet remains available in the sidebar.')
    else:
        if st.button('Generate a sentence'):
            old=st.session_state.pop('worker',None)
            if old:old.on_ended()
            st.session_state.prompt=' '.join(random.SystemRandom().choice(w) for _,w in SLOTS);st.session_state.take+=1;st.session_state.pop('reference',None)
        st.subheader(st.session_state.prompt)
        st.write('Read the sentence first. Start the camera, then speak as soon as the preview appears. The first three seconds make one take. Processing follows the take; stop the camera afterward.')
    if st.button('New take'):
        old=st.session_state.pop('worker',None)
        if old:old.on_ended()
        st.session_state.take+=1;st.session_state.pop('reference',None)
    if st.toggle('Enable webcam controls',value=False):
        from streamlit_webrtc import webrtc_streamer,WebRtcMode
        try:
            with st.spinner('Loading webcam model…'):model=chosen_reader(checkpoint_choice)
        except Exception as error:
            st.error(str(error));st.stop()
        if getattr(model,'selection_error',None):st.info('Verified fine-tuned weights are unavailable; using the original pretrained model.')
        def factory(reader=model, auto=is_auto):
            recognize=lambda frames:json.dumps(reader.predict_frames(frames)[0])
            return LiveSession(recognize,seconds=8) if auto else OneTakeSession(recognize)
        ctx=webrtc_streamer(key='exact-'+checkpoint_choice+'-'+str(st.session_state.take),mode=WebRtcMode.SENDRECV,rtc_configuration={'iceServers':[]},media_stream_constraints={'audio':False,'video':{'width':{'ideal':640 if is_auto else 360},'height':{'ideal':480 if is_auto else 288},'frameRate':{'ideal':25,'max':30}}},video_processor_factory=factory,async_processing=True,media_toggle_controls=False)
        if ctx.video_processor:
            st.session_state.worker=ctx.video_processor
            if not is_auto and 'reference' not in st.session_state:st.session_state.reference=st.session_state.prompt
        @st.fragment(run_every=.5)
        def poll():
            w=st.session_state.get('worker')
            if not w:return
            snap=w.snapshot()
            if snap['busy']:st.info('Reading mouth movements…')
            elif is_auto and not snap['closed']:st.caption(f"Collecting video: {snap['buffered_seconds']:.1f} / 8.0 seconds")
            if snap['notice']:st.warning(snap['notice'])
            for item in (snap['segments'][-12:] if is_auto else snap['segments'][:1]):
                if item.get('error'):st.error(item['error'])
                else:
                    if is_auto:st.caption(f"{item['start']:.1f}–{item['end']:.1f}s")
                    show(json.loads(item['text']),None if is_auto else st.session_state.get('reference'))
        poll()
    st.caption('Video only; no microphone or stored webcam video. Recorded-input checks are not a physical-camera accuracy test.')
with results:
    run=ROOT/'runs/paper-finetune-v1'
    if (run/'progress.json').exists():
        progress=json.loads((run/'progress.json').read_text())
        st.subheader('Original LipNet fine-tuning')
        st.write('Status: '+progress['stage'].replace('_',' '))
        if 'epoch' in progress:st.write(f"Epoch {progress['epoch']} / 6")
        if 'completed' in progress and 'total' in progress:st.progress(progress['completed']/progress['total'],text=f"{progress['completed']} / {progress['total']}")
        st.caption('Training speakers s3/s4; validation s1; final test s2. The reviewed model selector verifies checkpoint integrity and retains the original pretrained fallback.')
    if (run/'independent-review.json').exists():
        review=json.loads((run/'independent-review.json').read_text());m=review['metrics']
        st.table([{'Evaluation':'Validation (150 clips)','Pretrained WER':f"{m['baseline-validation']['metrics']['wer']:.2%}",'Fine-tuned WER':f"{m['selected-reload-validation']['metrics']['wer']:.2%}"},{'Evaluation':'Final test (200 clips)','Pretrained WER':f"{m['baseline-test']['metrics']['wer']:.2%}",'Fine-tuned WER':f"{m['candidate-test']['metrics']['wer']:.2%}"}])
        st.caption('Epoch 4 selected on validation before testing. The test improvement is modest and its paired 95% interval includes zero. One validation/test speaker each; general webcam accuracy is unproven.')
        st.caption('Both checkpoints use the same deterministic spelling tie-break. Historical source-restoration results below preceded this fix.')
        st.download_button('Download fine-tuning review',(run/'independent-review.json').read_text(),'lipnet-finetuning-review.json')
    for kind,label in [('tutorial','Notebook checkpoint — speaker 1 sample'),('paper','Paper unseen-speaker checkpoint — local sample')]:
        st.subheader(label);path=ROOT/'reports'/(kind+'-verification.json')
        if path.exists():
            v=json.loads(path.read_text());m=v.get('local_unseen_subset_metrics') or v['metrics']
            a,b=st.columns(2);a.metric('WER',f"{m['wer']:.2%}");b.metric('Exact sentences',f"{m['exact']} / {m['clips']}")
            st.caption('Published weights; no local fine-tuning. This sample is not the full official benchmark or a webcam test. The notebook checkpoint’s training split is unknown.')
            st.download_button('Download '+kind+' report',path.read_text(),kind+'-verification.json')
        else:st.info('Source-fidelity evaluation is running; no accuracy claim yet.')
