"""One-camera depth: a real hand shape posed at a known place must come back there."""

import urllib.request

import cv2
import numpy as np

from armtrack.vision.geometry import MIDDLE_MCP, WRIST
from armtrack.vision.hands import HandDetector
from armtrack.tracking.mono import MonoSolver, intrinsics


def test_pnp_recovers_position(tmp_path):
    p = tmp_path / "hands.jpg"
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg", p)
    d = HandDetector(1)
    _, _, world = d.detect_full(cv2.imread(str(p)), 0.0)[0]
    d.close()
    K, dist, _ = intrinsics((1280, 720), None, 2, 70.0)
    palm = 95.0
    obj = world * 1000 * palm / np.linalg.norm((world[MIDDLE_MCP] - world[WRIST]) * 1000)
    solver = MonoSolver(K, dist, palm)
    for t_true in ([0.0, 0.0, 500.0], [120.0, -60.0, 700.0], [-150.0, 80.0, 400.0]):
        rvec = np.array([0.3, -0.4, 0.2])
        px, _ = cv2.projectPoints(obj, rvec, np.array(t_true), K, dist)
        px = px.reshape(-1, 2) + np.random.default_rng(0).normal(0, 0.5, (21, 2))  # half-pixel noise
        X, err = solver.solve(px, world)
        X_true = obj @ cv2.Rodrigues(rvec)[0].T + t_true
        assert err < 2.0
        assert np.abs(X - X_true).max() < 0.04 * t_true[2], np.abs(X - X_true).max()

    # an unmeasured hand 10% bigger than the model reads about 10% too far away
    solver = MonoSolver(K, dist, palm / 1.1)
    solver.prev = None
    px, _ = cv2.projectPoints(obj, np.array([0.3, -0.4, 0.2]), np.array([0.0, 0, 500]), K, dist)
    X, _ = solver.solve(px.reshape(-1, 2), world)
    assert 430 < X[:, 2].mean() < 480
