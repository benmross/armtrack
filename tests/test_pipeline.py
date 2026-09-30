"""End-to-end checks with two virtual cameras, so the math is verified without hardware.

The board is rendered into two simulated cameras ~40 cm apart looking at the
same space from different angles (the desk setup), calibrated with the real
code, and then a synthetic hand is triangulated and its pose recovered.
"""

import cv2
import numpy as np

from stereo_hand.board import Detector, make_board
from stereo_hand.calib import desk_frame, solve
from stereo_hand.geometry import hand_pose, match_hands, to_world, triangulate

SQ = 30.0
W, H = 1280, 720
K = np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]])
D = np.zeros(5)


def look_at(eye, target):
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, -1, 0])  # world up is -y... keep image y pointing down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world -> camera
    return R, -R @ eye


# rig: cam1 on the left of the desk, cam2 on the right, both aimed at a point 50 cm out
C1 = look_at(np.array([-200.0, 0, 0]), np.array([0.0, 0, 500]))
C2 = look_at(np.array([200.0, 0, 0]), np.array([0.0, 0, 500]))


def render_board(board_img, board_px_per_mm, R_bw, t_bw, cam):
    """Warp the flat board image into a camera, given the board's pose in world."""
    R, t = cam
    corners_mm = np.array([[0, 0, 0], [7 * SQ, 0, 0], [7 * SQ, 5 * SQ, 0], [0, 5 * SQ, 0]], float)
    world = corners_mm @ R_bw.T + t_bw
    cam_pts = world @ R.T + t
    img_pts = (cam_pts @ K.T)
    img_pts = img_pts[:, :2] / img_pts[:, 2:]
    m = 40  # margin in board image px
    src = np.float32([[m, m], [m + 7 * SQ * board_px_per_mm, m],
                      [m + 7 * SQ * board_px_per_mm, m + 5 * SQ * board_px_per_mm],
                      [m, m + 5 * SQ * board_px_per_mm]])
    Hm = cv2.getPerspectiveTransform(src, img_pts.astype(np.float32))
    out = cv2.warpPerspective(board_img, Hm, (W, H), borderValue=200)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def rot(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


def test_calibration_recovers_rig():
    ppm = 6.0
    board = make_board(SQ)
    img = board.generateImage((int(7 * SQ * ppm) + 80, int(5 * SQ * ppm) + 80), marginSize=40)
    det = Detector(SQ)
    rng = np.random.default_rng(0)
    mono1, mono2, pairs = [], [], []
    for _ in range(30):
        R_bw = rot(*rng.uniform(-0.5, 0.5, 3))
        t_bw = np.array([rng.uniform(-150, 0), rng.uniform(-120, 20), rng.uniform(400, 650)])
        f1 = render_board(img, ppm, R_bw, t_bw, C1)
        f2 = render_board(img, ppm, R_bw, t_bw, C2)
        r1, r2 = det.detect(f1), det.detect(f2)
        if r1 is not None:
            mono1.append(r1)
        if r2 is not None:
            mono2.append(r2)
        if r1 is not None and r2 is not None:
            pairs.append((r1, r2))
    assert len(pairs) >= 10, len(pairs)
    cal = solve(det, mono1, mono2, pairs, (W, H), (W, H), SQ)
    assert abs(cal.baseline_mm - 400) < 8, cal.baseline_mm
    assert cal.rms < 1.0
    R_true = C2[0] @ C1[0].T
    assert np.degrees(np.arccos(np.clip((np.trace(cal.R.T @ R_true) - 1) / 2, -1, 1))) < 1.0

    # a synthetic hand 50 cm out, projected into both cameras and triangulated back
    hand_w = rng.uniform([-60, -60, 450], [60, 60, 560], size=(21, 3))
    hand_w[0] = [0, 40, 500]            # wrist
    hand_w[9] = [0, -40, 500]           # middle knuckle: fingers point "up" (-y)
    hand_w[5], hand_w[17] = [25, -35, 500], [-25, -35, 500]
    hand_w[4], hand_w[8] = [30, -60, 500], [30, -90, 500]

    def proj(cam, X):
        R, t = cam
        p = (X @ R.T + t) @ K.T
        return p[:, :2] / p[:, 2:]

    px1, px2 = proj(C1, hand_w), proj(C2, hand_w)
    # a decoy: a different hand in cam2 only, which must not be paired
    decoy = proj(C2, hand_w + [150, 0, 0])
    got = match_hands(cal, [("Right", px1)], [("Left", decoy), ("Right", px2)])
    assert len(got) == 1 and got[0][2] < 2.0
    Xc = got[0][1]
    X_true_c1 = hand_w @ C1[0].T + C1[1]
    # within 1% of range; the residual comes from resampling the rendered board
    assert np.abs(Xc - X_true_c1).max() < 6.0, np.abs(Xc - X_true_c1).max()
    pose = hand_pose(Xc)
    assert abs(pose["pinch_mm"] - 30) < 3

    # desk frame from a board lying face-up on a desk 20 cm below the cameras
    R_bw = rot(-np.pi / 2, 0, 0)
    t_bw = np.array([-100.0, 200, 650])
    f1 = render_board(img, ppm, R_bw, t_bw, C1)
    ids, pts = det.detect(f1)
    ok, rv, tv = cv2.solvePnP(det.all_obj[ids], pts, cal.K1, cal.d1)
    cal.Rw, cal.tw = desk_frame(cv2.Rodrigues(rv)[0], tv.ravel())
    corner0_cam1 = t_bw @ C1[0].T + C1[1]
    assert np.abs(to_world(cal, corner0_cam1[None])[0]).max() < 6.0  # board origin is world origin
    up_world = to_world(cal, (corner0_cam1 + C1[0] @ np.array([0, -100.0, 0]))[None])[0]
    assert up_world[2] > 95  # 10 cm "up" in the scene (-y) is +z on the desk


def test_mediapipe_detects_sample(tmp_path):
    import urllib.request
    from stereo_hand.hands import HandDetector
    p = tmp_path / "hands.jpg"
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg", p)
    d = HandDetector(2)
    hands = d.detect(cv2.imread(str(p)), 0.0)
    d.close()
    assert len(hands) == 2 and hands[0][1].shape == (21, 2)
