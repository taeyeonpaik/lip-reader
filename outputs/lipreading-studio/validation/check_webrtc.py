"""Exercise local WebRTC transport with recorded video, without camera access.

Run from the app directory: python validation/check_webrtc.py
Requires the same environment and included weights as the app.
"""
import asyncio
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
from aiortc import RTCPeerConnection, RTCConfiguration, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from streamlit_webrtc.process import AsyncVideoProcessTrack
from studio.live import LiveSession
from studio.pipeline import ROOT, LipReader, track_mouth


class RecordedCamera(VideoStreamTrack):
    def __init__(self, path):
        super().__init__()
        with av.open(str(path)) as container:
            self.frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        self.index = 0
        self.started = None

    async def recv(self):
        if self.started is None:
            self.started = time.monotonic()
        await asyncio.sleep(max(0, self.started + self.index / 25 - time.monotonic()))
        if self.index >= len(self.frames) + 5:
            raise MediaStreamError
        frame = av.VideoFrame.from_ndarray(self.frames[min(self.index, len(self.frames)-1)], format="rgb24")
        frame.pts = self.index * 3600
        frame.time_base = Fraction(1, 90000)
        self.index += 1
        return frame


async def main():
    reader = LipReader()
    live = LiveSession(lambda frames: reader.recognize(track_mouth(frames)[0]))
    sender, receiver = (RTCPeerConnection(RTCConfiguration(iceServers=[])) for _ in range(2))
    received = asyncio.Event()
    processors = []
    tasks = []
    frames_received = 0
    sender.addTrack(RecordedCamera(ROOT / "examples/research-demo.mp4"))

    @receiver.on("track")
    def got_track(track):
        assert track.kind == "video"
        processed = AsyncVideoProcessTrack(track, live)
        processors.append(processed)
        async def consume():
            nonlocal frames_received
            while frames_received < 200:
                await processed.recv()
                frames_received += 1
            # Let the async processor drain its last queued frame before stopping.
            await asyncio.sleep(.2)
            received.set()
        tasks.append(asyncio.create_task(consume()))

    started = time.monotonic()
    try:
        await sender.setLocalDescription(await sender.createOffer())
        await receiver.setRemoteDescription(sender.localDescription)
        await receiver.setLocalDescription(await receiver.createAnswer())
        await sender.setRemoteDescription(receiver.localDescription)
        assert "m=audio" not in sender.localDescription.sdp
        await asyncio.wait_for(received.wait(), 45)
        for processed in processors:
            processed.stop()
        assert await asyncio.to_thread(live.wait_until_idle, 60)
        state = live.snapshot()
        assert state["closed"] and state["segments"] and state["skipped"] == 0, state
        assert all(s["text"] and not s.get("error") for s in state["segments"]), state
        report = {"transport": "two local aiortc peers, VP8 video over WebRTC",
                  "physical_camera_tested": False, "audio_tracks": 0,
                  "frames_received": frames_received,
                  "seconds_including_capture": time.monotonic()-started, **state}
        (ROOT/"validation/webcam-webrtc-test.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        for task in tasks:
            task.cancel()
        await sender.close()
        await receiver.close()
        live.on_ended()


if __name__ == "__main__":
    asyncio.run(main())
