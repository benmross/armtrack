"""MediaPipe hand landmarks, one detector per camera (each keeps its own tracking state)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading hand model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandDetector:
    def __init__(self, num_hands: int = 2, min_conf: float = 0.5):
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, frame: np.ndarray, t: float) -> list[tuple[str, np.ndarray]]:
        """Returns [(handedness label, (21,2) pixel coords)]."""
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ms)
        h, w = frame.shape[:2]
        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)
            out.append((hd[0].category_name, pts))
        return out

    def close(self) -> None:
        self.lm.close()
