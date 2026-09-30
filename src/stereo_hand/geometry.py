"""Triangulation, matching a hand across views, smoothing, and the hand's pose."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .calib import Calibration

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def triangulate(cal: Calibration, px1: np.ndarray, px2: np.ndarray) -> np.ndarray:
    """(N,2) pixel points in each camera -> (N,3) points in camera-1 coordinates, mm."""
    n1 = cv2.undistortPoints(px1.reshape(-1, 1, 2).astype(np.float64), cal.K1, cal.d1).reshape(-1, 2)
    n2 = cv2.undistortPoints(px2.reshape(-1, 1, 2).astype(np.float64), cal.K2, cal.d2).reshape(-1, 2)
    P1, P2 = cal.projections()
    X = cv2.triangulatePoints(P1, P2, n1.T, n2.T)
    return (X[:3] / X[3]).T


def reprojection_error(cal: Calibration, X: np.ndarray, px1: np.ndarray, px2: np.ndarray) -> float:
    """Mean pixel distance between the observed points and the 3-D points projected back."""
    z = np.zeros(3)
    r1, _ = cv2.projectPoints(X, z, z, cal.K1, cal.d1)
    rvec2, _ = cv2.Rodrigues(cal.R)
    r2, _ = cv2.projectPoints(X, rvec2, cal.T, cal.K2, cal.d2)
    e1 = np.linalg.norm(r1.reshape(-1, 2) - px1, axis=1)
    e2 = np.linalg.norm(r2.reshape(-1, 2) - px2, axis=1)
    return float((e1.mean() + e2.mean()) / 2)


def match_hands(cal: Calibration, hands1: list, hands2: list, want: str | None = None,
                max_err: float = 30.0):
    """Pair up hands seen in the two cameras. Handedness labels are unreliable
    from a side view, so a pair is chosen by which triangulation agrees best with
    both images. Returns a list of (label, points_mm (21,3), reprojection error)."""
    cands = []
    for i, (lab1, p1) in enumerate(hands1):
        for j, (lab2, p2) in enumerate(hands2):
            X = triangulate(cal, p1, p2)
            if np.median(X[:, 2]) <= 0:  # behind camera 1: not a real pairing
                continue
            err = reprojection_error(cal, X, p1, p2)
            if err <= max_err:
                cands.append((err, i, j, lab1 if lab1 == lab2 else lab1, X))
    cands.sort(key=lambda c: c[0])
    used1, used2, out = set(), set(), []
    for err, i, j, lab, X in cands:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        out.append((lab, X, err))
    if want:
        out = [h for h in out if h[0].lower() == want.lower()] or out[:1]
    return out


def to_world(cal: Calibration, X: np.ndarray) -> np.ndarray:
    return X @ cal.Rw.T + cal.tw


class OneEuro:
    """One Euro filter (Casiez et al. 2012): smooth when still, responsive when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None or t <= self.t or t - self.t > 0.5:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def hand_pose(X: np.ndarray) -> dict:
    """What an arm controller wants: where the hand is, which way it faces, and
    how closed the grip is. Rotation columns are the hand's x (toward the index
    side), y (wrist toward fingers) and z (out of the palm) axes."""
    wrist = X[WRIST]
    fwd = X[MIDDLE_MCP] - wrist
    across = X[INDEX_MCP] - X[PINKY_MCP]
    y = fwd / (np.linalg.norm(fwd) + 1e-9)
    z = np.cross(across, y)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    palm = float(np.linalg.norm(fwd))
    pinch = float(np.linalg.norm(X[THUMB_TIP] - X[INDEX_TIP]))
    # 0 = pinched shut, 1 = wide open, normalised by palm length so it is size-independent
    grip = float(np.clip((pinch / (palm + 1e-9) - 0.15) / (1.2 - 0.15), 0, 1))
    return {
        "wrist_mm": wrist.tolist(),
        "palm_center_mm": X[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]].mean(0).tolist(),
        "rotation": R.tolist(),
        "quat_wxyz": rot_to_quat(R).tolist(),
        "pinch_mm": pinch,
        "grip_open": grip,
        "palm_length_mm": palm,
    }


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])
