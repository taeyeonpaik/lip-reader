"""Local GRID LipNet practice studio. No audio or reference enters recognition."""
from pathlib import Path
import hashlib,json,tempfile
import streamlit as st
from tutorial.runtime import ROOT,GridReader,read_rgb,prepare_upload
from tutorial.grammar import SLOTS,generate_sentence,word_error
from tutorial.live import OneTakeSession

st.set_page_config(page_title='LipNet · GRID practice',page_icon='💬',layout='wide')
st.title('LipNet · Read a six-word sentence')
st.caption('The original tutorial model · constrained English · local processing')
st.write('Use the GRID sentence pattern below. Speak naturally while keeping your face visible; the app uses video only.')

@st.cache_resource
def reader_for_selection(checkpoint,mtime):return GridReader(checkpoint)
def reader():
    selection=ROOT/'models/grid-selection.json'
    chosen=json.loads(selection.read_text()) if selection.exists() else {}
    path=ROOT/'models/grid-adapted.pt'
    if chosen.get('selected')=='adapted':
        from tutorial.runtime import digest
        if not path.exists() or digest(path)!=chosen['sha256']:raise RuntimeError('Selected model integrity check failed.')
        return reader_for_selection(str(path),path.stat().st_mtime_ns)
    return reader_for_selection(None,0)

with st.sidebar:
    st.header('Recording guide')
    st.write('One speaker, facing the camera, in good light. Keep your head steady and your whole face visible. Use normal articulation without exaggerating.')
    st.write('One six-word sentence per take, ideally about three seconds. Uploads may be 1–5 seconds.')
    st.caption('No microphone. No webcam recordings are saved. Transcripts remain in this browser session until cleared.')
    st.warning('Experimental: held-out GRID speaker WER is 81.25% with automatic alignment. This is not a reliable webcam lip reader. Grammar-valid predictions can still be wrong.')
    st.write('The practice sentence is used only for scoring afterward. It is never supplied to the recognizer.')
    st.link_button('Original video','https://www.youtube.com/watch?v=uKyojQjbx4c')
    st.link_button('Tutorial source','https://github.com/nicknochnack/LipNet')
    st.link_button('Paper implementation','https://github.com/rizkiarm/LipNet')

st.subheader('The sentence pattern')
st.markdown('**Command → Color → Preposition → Letter → Number → Ending**')
with st.expander('Show every allowed word',expanded=True):
    st.table([{'Position':name,'Choose one':', '.join(words).upper() if name=='Letter' else ', '.join(words)} for name,words in SLOTS])
st.caption('Letters exclude W. Say the letter name and the number as a word—for example, “bin red at S nine again.”')

practice,upload,examples,results=st.tabs(['Webcam practice','Upload a sentence','GRID examples','Training & results'])
def show_result(result,reference=None):
    st.subheader('Recognized sentence')
    st.write(result['constrained'] or 'No complete GRID sentence decoded. Try again.')
    st.caption('Grammar-constrained CTC decoding: searches all valid GRID sentences, not just the practice prompt.')
    st.write('**Unconstrained model output:** '+(result['raw'] or '(empty)'))
    st.caption(f"{result['seconds']:.2f}s processing · {result['model']} · {result['mode']}")
    if reference:
        st.write('**Sentence shown for this take:** '+reference)
        measured=word_error(reference,result['constrained'])
        a,b=st.columns(2);a.metric('Word errors',f"{measured['errors']} / {measured['words']}");b.metric('This take WER',f"{measured['wer']:.1%}")
        st.caption('This score assumes you said the displayed sentence exactly. It is practice feedback, not a benchmark.')

with practice:
    st.subheader('Practice one sentence')
    if 'prompt' not in st.session_state:st.session_state.prompt=generate_sentence()
    if 'take_number' not in st.session_state:st.session_state.take_number=0
    if st.button('Generate a random sentence'):
        old=st.session_state.pop('last_take',None)
        if old:old.on_ended()
        st.session_state.prompt=generate_sentence();st.session_state.take_number+=1
        st.session_state.pop('take_reference',None)
    st.markdown('### '+st.session_state.prompt)
    st.write('Read it first. Click Start, allow the camera, then say the sentence as soon as the preview appears. The first three seconds form one take. Click Stop afterward; use New take to retry.')
    if st.button('New take with this sentence'):
        old=st.session_state.pop('last_take',None)
        if old:old.on_ended()
        st.session_state.take_number+=1;st.session_state.pop('take_reference',None)
    from streamlit_webrtc import webrtc_streamer,WebRtcMode
    live_model=reader()
    def make_take():
        return OneTakeSession(lambda frames:json.dumps(live_model.recognize(frames)))
    context=webrtc_streamer(key='grid-take-'+str(st.session_state.take_number),mode=WebRtcMode.SENDRECV,rtc_configuration={'iceServers':[]},media_stream_constraints={'audio':False,'video':{'width':{'ideal':640},'height':{'ideal':480},'frameRate':{'ideal':25,'max':30},'facingMode':'user'}},video_processor_factory=make_take,async_processing=True,media_toggle_controls=False)
    if context.video_processor is not None:
        st.session_state.last_take=context.video_processor
        if 'take_reference' not in st.session_state:st.session_state.take_reference=st.session_state.prompt
    @st.fragment(run_every=.5)
    def take_results():
        take=st.session_state.get('last_take')
        if take is None:st.info('Camera off. Get ready, then click Start.');return
        state=take.snapshot()
        if state['busy']:st.info('Reading your take…')
        elif not state['segments']:st.info(f"Capturing: {state['buffered_seconds']:.1f} / 3.0 seconds")
        if state['notice']:st.warning(state['notice'])
        if state['segments']:
            segment=state['segments'][0]
            if segment.get('error'):st.error(segment['error'])
            else:
                result=json.loads(segment['text']);show_result(result,st.session_state.get('take_reference'))
                st.download_button('Download practice result',json.dumps({'prediction':result,'reference_for_scoring_only':st.session_state.get('take_reference')},indent=2),'grid-practice.json','application/json')
            st.caption('Take complete. Stop the camera before starting another take.')
    take_results()
with upload:
    file=st.file_uploader('Upload one 1–5 second sentence',type=['mp4','mov','webm','mpg','avi'],max_upload_size=50)
    reference=st.text_input('Optional transcript for scoring afterward',help='This is never passed into recognition.')
    if file and st.button('Read uploaded video',type='primary'):
        try:
            with st.spinner('Tracking the face and reading the sentence…'),tempfile.TemporaryDirectory(prefix='grid-upload-') as folder:
                source=Path(folder)/('source'+Path(file.name).suffix);source.write_bytes(file.getvalue());normalized=Path(folder)/'silent.mp4'
                prepare_upload(source,normalized);result=reader().recognize(read_rgb(normalized))
            st.session_state.upload_result=(hashlib.sha256(file.getvalue()).hexdigest(),result)
        except Exception as e:st.error(str(e))
    saved=st.session_state.get('upload_result')
    if file and saved and saved[0]==hashlib.sha256(file.getvalue()).hexdigest():show_result(saved[1],reference)
with examples:
    paths=sorted((ROOT/'samples').glob('*/*.mpg'))
    selected=st.selectbox('Original tutorial example',paths,format_func=lambda p:p.stem)
    st.video(str(selected.with_suffix('.mp4')))
    st.caption('These examples use the original fixed crop. They are demonstrations, not independent tests of the author checkpoint.')
    if st.button('Read GRID example'):
        with st.spinner('Reading…'):result=reader().recognize(read_rgb(selected),fixed=True)
        text=' '.join(line.split()[-1] for line in selected.with_suffix('.align').read_text().splitlines() if line.split()[-1] not in ['sil','sp'])
        show_result(result,text)
with results:
    st.subheader('Accuracy on new speakers')
    st.write('The model is the tutorial’s 3D CNN and two bidirectional LSTMs. It starts from the author’s published 96-epoch checkpoint. The original TensorFlow reproduction is preserved alongside the verified Mac GPU adapter.')
    st.caption('The linked paper implementation uses GRUs and is a separate architecture; we do not label this LSTM tutorial model an exact paper reproduction.')
    @st.fragment(run_every=10)
    def progress():
        run=ROOT/'runs/grid-return-v1'
        result=run/'result.json';failure=run/'failure.json';status=run/'progress.json'
        if result.exists():
            report=json.loads(result.read_text())
            st.warning('Adaptation finished without a qualifying gain. The author checkpoint remains active.')
            a,b=st.columns(2)
            a.metric('Held-out speaker WER · grammar',f"{report['selected_test']['constrained']['wer']:.2%}")
            b.metric('Held-out speaker WER · raw',f"{report['selected_test']['raw']['wer']:.2%}")
            st.caption('200 clips from speaker 4 with automatic alignment; zero exact sentences. This is not the original paper benchmark or a physical-webcam evaluation.')
            with st.expander('Full experiment report'):st.json(report)
            st.download_button('Download evaluation report',result.read_text(),'grid-results.json','application/json')
        elif failure.exists():st.error(json.loads(failure.read_text())['error'])
        elif status.exists():st.info('Local experiment in progress');st.json(json.loads(status.read_text()))
        else:st.info('Preparing additional GRID speakers. No new-speaker WER result yet.')
    progress()
    st.write('Training: speakers 1 and 3. Validation: speaker 2. Final test: speaker 4, untouched until model selection. Both unconstrained and grammar-constrained WER are reported.')
    st.warning('Speakers are disjoint in our local experiment. The tutorial checkpoint’s complete original training history is not published, so we cannot certify that these speakers were absent from its pretraining. Dataset performance also does not establish accuracy on your webcam.')
