"""Stereo calibration: each camera's lens, then where the cameras are relative to each other."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from armtrack.vision.board import Detector


@dataclass
class Calibration:
    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray  # camera-1 coords -> camera-2 coords
    T: np.ndarray  # in mm
    size1: tuple[int, int]
    size2: tuple[int, int]
    rms: float
    square_mm: float
    # desk frame: world = Rw @ cam1 + tw. Identity until `w` is pressed in track.
    Rw: np.ndarray = field(default_factory=lambda: np.eye(3))
    tw: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    def projections(self) -> tuple[np.ndarray, np.ndarray]:
        """Projection matrices for *normalised* (undistorted, K-removed) coordinates."""
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([self.R, self.T.reshape(3, 1)])
        return P1, P2

    def save(self, path: str | Path) -> None:
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        Path(path).write_text(json.dumps(out, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        raw = json.loads(Path(path).read_text())
        for k in ("K1", "d1", "K2", "d2", "R", "T", "Rw", "tw"):
            raw[k] = np.array(raw[k], dtype=np.float64)
        raw["size1"], raw["size2"] = tuple(raw["size1"]), tuple(raw["size2"])
        return cls(**raw)


Sample = tuple[np.ndarray, np.ndarray]  # (ids, image points)


def _mono(obj_all: np.ndarray, samples: list[Sample], size: tuple[int, int]):
    objs = [obj_all[ids].reshape(-1, 1, 3) for ids, _ in samples]
    imgs = [pts.reshape(-1, 1, 2) for _, pts in samples]
    rms, K, d, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return rms, K, d


def solve(det: Detector, mono1: list[Sample], mono2: list[Sample], pairs: list[tuple[Sample, Sample]],
          size1: tuple[int, int], size2: tuple[int, int], square_mm: float) -> Calibration:
    """mono lists: every view of the board per camera. pairs: simultaneous views in both."""
    if len(pairs) < 5:
        raise ValueError(f"need at least 5 views seen by both cameras, have {len(pairs)}")
    rms1, K1, d1 = _mono(det.all_obj, mono1, size1)
    rms2, K2, d2 = _mono(det.all_obj, mono2, size2)
    objs, im1, im2 = [], [], []
    for (ids1, p1), (ids2, p2) in pairs:
        common, i1, i2 = np.intersect1d(ids1, ids2, return_indices=True)
        if len(common) < 6:
            continue
        objs.append(det.all_obj[common].reshape(-1, 1, 3))
        im1.append(p1[i1].reshape(-1, 1, 2))
        im2.append(p2[i2].reshape(-1, 1, 2))
    if len(objs) < 5:
        raise ValueError("too few views where both cameras saw the same corners")
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)
    # size is only used for initialising intrinsics, which are fixed here
    rms, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        objs, im1, im2, K1, d1, K2, d2, size1, criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"lens fit: camera 1 {rms1:.2f}px, camera 2 {rms2:.2f}px; stereo fit {rms:.2f}px")
    return Calibration(K1, d1, K2, d2, R, T.ravel(), size1, size2, float(rms), square_mm)


def board_pose(det: Detector, frame: np.ndarray, K: np.ndarray, d: np.ndarray):
    """Pose of the board in the camera. Returns (R, t) board->camera or None."""
    found = det.detect(frame)
    if found is None:
        return None
    ids, pts = found
    ok, rvec, tvec = cv2.solvePnP(det.all_obj[ids], pts, K, d)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def desk_frame(R_bc: np.ndarray, t_bc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World frame from a board lying flat on the desk: x along the board's long
    edge, y across it, z up out of the desk. Returns (Rw, tw) with world = Rw @ cam + tw."""
    flip = np.diag([1.0, -1.0, -1.0])  # board y points down the page and z into the desk
    Rw = flip @ R_bc.T
    tw = -Rw @ t_bc
    return Rw, tw
