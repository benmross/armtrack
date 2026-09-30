"""Camera sources: a live camera by index, or a video file / URL.

Live cameras are read on their own thread and only the newest frame is kept,
so a slow detector never works on a stale, queued-up frame. The two cameras
are not hardware-synchronised; the timestamps say how far apart a pair is.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


def _parse(src: str | int) -> str | int:
    if isinstance(src, str) and src.isdigit():
        return int(src)
    return src


def open_capture(src: str | int, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    src = _parse(src)
    if isinstance(src, int) and sys.platform == "darwin":
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {src!r}. Run `stereo-hand cameras` to see which "
            "indices exist, and check the terminal has camera permission."
        )
    return cap


class Source:
    """Newest-frame-wins reader. `read()` returns (frame, timestamp) or (None, None)."""

    def __init__(self, src: str | int, width: int = 1280, height: int = 720):
        self.src = _parse(src)
        self.live = isinstance(self.src, int) or str(self.src).startswith(("http", "rtsp"))
        self.cap = open_capture(self.src, width, height)
        self._frame: np.ndarray | None = None
        self._t: float | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self._pending: np.ndarray | None = None
        if not self.live:
            ok, self._pending = self.cap.read()
            if not ok:
                raise SystemExit(f"Could not read a frame from {self.src!r}.")
            self._frame = self._pending
        if self.live:
            threading.Thread(target=self._run, daemon=True).start()
            deadline = time.time() + 5
            while self._frame is None and time.time() < deadline:
                time.sleep(0.01)
            if self._frame is None:
                raise SystemExit(f"Camera {self.src!r} opened but produced no frames.")
        self.size = (self._frame.shape[1], self._frame.shape[0])

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame, self._t = frame, time.time()
                self._seq += 1

    def read(self) -> tuple[np.ndarray | None, float | None]:
        if not self.live:
            if self._pending is not None:
                frame, self._pending = self._pending, None
                return frame, time.time()
            ok, frame = self.cap.read()
            return (frame, time.time()) if ok else (None, None)
        with self._lock:
            return self._frame, self._t

    def close(self) -> None:
        self._stop = True
        self.cap.release()


def probe(max_index: int = 6) -> list[tuple[int, np.ndarray]]:
    """Open every camera index that works and grab one frame from each.
    Indices are contiguous, so stop at the first one that does not open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        frame = None
        for _ in range(20):  # the first frames from some cameras are black
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is not None:
            found.append((i, frame))
    return found


def contact_sheet(frames: list[tuple[int, np.ndarray]], tile_w: int = 480) -> np.ndarray:
    tiles = []
    for i, f in frames:
        h = int(f.shape[0] * tile_w / f.shape[1])
        t = cv2.resize(f, (tile_w, h))
        label = f"index {i}  {f.shape[1]}x{f.shape[0]}"
        cv2.rectangle(t, (0, 0), (tile_w, 40), (0, 0, 0), -1)
        cv2.putText(t, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    return np.hstack(tiles)


def preview(indices: list[int], width: int = 1280, height: int = 720) -> None:
    """Live view of every camera, labelled with its index, until a key is pressed."""
    sources = [(i, Source(i, width, height)) for i in indices]
    try:
        while True:
            frames = [(i, s.read()[0]) for i, s in sources]
            cv2.imshow("cameras: live (any key to close)", contact_sheet([(i, f) for i, f in frames if f is not None]))
            if cv2.waitKey(1) & 0xFF != 255:
                break
    finally:
        for _, s in sources:
            s.close()
        cv2.destroyAllWindows()
