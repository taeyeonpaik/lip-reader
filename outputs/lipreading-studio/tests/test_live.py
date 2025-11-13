import threading
import asyncio
from fractions import Fraction
import av
import numpy as np
from studio.live import LiveSession


def test_queued_webrtc_frames_preserve_all_samples_and_stop_flushes():
    sizes=[]
    live=LiveSession(lambda frames: sizes.append(len(frames)) or "words",seconds=1)
    queued=[]
    for i in range(25):
        frame=av.VideoFrame.from_ndarray(np.zeros((16,16,3),np.uint8),format="rgb24")
        frame.pts=i*3600
        frame.time_base=Fraction(1,90000)
        queued.append(frame)
    previews=asyncio.run(live.recv_queued(queued))
    live.on_ended()
    live.thread.join(2)
    assert previews==[queued[-1]]
    assert sizes==[25]
    assert live.snapshot()["segments"][0]["end"]==1


def test_resamples_30fps_and_flushes_final_short_segment():
    sizes = []
    def recognize(frames):
        sizes.append(len(frames))
        return "words"
    live = LiveSession(recognize, seconds=1)
    frame = np.zeros((32,32,3), np.uint8)
    for index in range(60):
        live.accept(frame, index/30)
        if index == 30:
            assert live.wait_until_idle()
    live.on_ended()
    assert live.wait_until_idle()
    live.thread.join(2)
    assert sizes == [25,25]
    assert live.snapshot()["segments"][-1]["end"] == 2
    assert not live.thread.is_alive()


def test_camera_gap_does_not_duplicate_a_long_missing_interval():
    live = LiveSession(lambda frames: "words")
    frame = np.zeros((32,32,3), np.uint8)
    live.accept(frame, 0)
    live.accept(frame, 5)
    state = live.snapshot()
    assert state["buffered_seconds"] == .04
    assert state["skipped"] == 1
    live.on_ended()
    live.thread.join(2)
    assert not live.snapshot()["segments"]


def test_slow_inference_has_bounded_queue_and_reports_skips():
    entered, release = threading.Event(), threading.Event()
    def recognize(frames):
        entered.set()
        assert release.wait(5)
        return "words"
    live = LiveSession(recognize, seconds=1)
    frame = np.zeros((16,16,3), np.uint8)
    try:
        for i in range(25):
            live.accept(frame,i/25)
        assert entered.wait(2)
        for i in range(25,100):
            live.accept(frame,i/25)
        state = live.snapshot()
        assert state["queued"] == 1 and state["skipped"] == 2
    finally:
        release.set()
        live.on_ended()
        live.thread.join(5)
    assert [(round(s["start"],3),round(s["end"],3)) for s in live.snapshot()["segments"]] == [(0,1),(3,4)]


def test_failure_is_visible_and_stop_prevents_new_frames():
    def fail(frames):
        raise ValueError("No face found")
    live = LiveSession(fail, seconds=1)
    frame = np.zeros((16,16,3), np.uint8)
    for i in range(25):
        live.accept(frame,i/25)
    live.on_ended()
    live.thread.join(2)
    state = live.snapshot()
    assert state["segments"][0]["error"] == "No face found"
    live.accept(frame,2)
    assert live.snapshot()["buffered_seconds"] == 0
