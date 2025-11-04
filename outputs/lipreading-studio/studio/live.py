"""Bounded, thread-safe webcam buffering; no audio capture or disk recordings."""
from collections import deque
import threading
import time

import cv2
import numpy as np


class LiveSession:
    def __init__(self, process_frames, seconds=8, clock=time.monotonic):
        self.process_frames = process_frames
        self.seconds = seconds
        self.clock = clock
        self.condition = threading.Condition()
        self.pending = deque()
        self.results = deque(maxlen=300)
        self.frames = []
        self.origin = self.next_sample = self.last_timestamp = None
        self.chunk_start = 0.0
        self.closed = False
        self.busy = False
        self.skipped = 0
        self.error = ""
        self.thread = threading.Thread(target=self._work, name="lipreading-live", daemon=True)
        self.thread.start()

    def recv(self, frame):
        self.accept(frame.to_ndarray(format="rgb24"), frame.time)
        return frame

    async def recv_queued(self, frames):
        for frame in frames:
            self.recv(frame)
        return [frames[-1]]

    def accept(self, rgb, timestamp=None):
        timestamp = self.clock() if timestamp is None else float(timestamp)
        height, width = rgb.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)))
        with self.condition:
            if self.closed:
                return
            if self.origin is None:
                self.origin = self.next_sample = timestamp
            self.last_timestamp = timestamp
            while self.next_sample <= timestamp + 1e-7:
                if not self.frames:
                    self.chunk_start = self.next_sample - self.origin
                self.frames.append(rgb.copy())
                self.next_sample += 1 / 25
                if len(self.frames) == round(self.seconds * 25):
                    self._enqueue()

    def _enqueue(self):
        if not self.frames:
            return
        # Keep only one waiting segment in addition to the one being processed.
        if self.pending:
            self.pending.clear()
            self.skipped += 1
            self.error = "Processing fell behind; an older waiting segment was skipped."
        end = self.chunk_start + len(self.frames) / 25
        self.pending.append((self.chunk_start, end, np.stack(self.frames)))
        self.frames.clear()
        self.condition.notify_all()

    def on_ended(self):
        with self.condition:
            if self.closed:
                return
            self.frames.clear()
            self.closed = True
            self.condition.notify_all()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closed)
                if not self.pending:
                    return
                start, end, frames = self.pending.popleft()
                self.busy = True
            begun = self.clock()
            try:
                text = self.process_frames(frames)
                result = {"start": start, "end": end, "text": text,
                          "processing_seconds": self.clock() - begun}
            except Exception as error:
                result = {"start": start, "end": end, "text": "", "error": str(error)}
            with self.condition:
                self.results.append(result)
                self.busy = False
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"segments": [dict(r) for r in self.results], "busy": self.busy,
                    "queued": len(self.pending), "buffered_seconds": len(self.frames) / 25,
                    "closed": self.closed, "skipped": self.skipped, "notice": self.error}

    def wait_until_idle(self, timeout=30):
        with self.condition:
            return self.condition.wait_for(lambda: not self.pending and not self.busy, timeout)
