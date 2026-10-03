"""Hand position in 3-D from one camera.

One camera cannot see depth directly. What it can do is compare how big the
hand looks with how big it really is. MediaPipe supplies a metric 3-D model of
the hand's shape (in its own frame, sized for an average hand); solvePnP then
finds the rotation and translation that make that model line up with the 2-D
landmarks in the image, given the camera's lens. The translation is the hand's
position in the camera's frame.

Depth is therefore only as good as the hand-size assumption: a hand 10% bigger
than the model reads 10% too far away. Measure your palm and pass --palm-mm to
remove most of that. Sideways and up/down motion is far more precise than depth,
which is the honest trade against two cameras.
"""

from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from .board import Detector
from .calib import Calibration, board_pose, desk_frame
from .cameras import Source
from .geometry import MIDDLE_MCP, WRIST, OneEuro, hand_pose
from .hands import HandDetector
from .arm import ArmController
from .viewer import Viewer
from .viz import View3D, draw_2d


def intrinsics(size: tuple[int, int], calib: str | None, calib_cam: int, hfov: float):
    """Lens from a stereo calibration if one exists for this camera, else a guess from the field of view."""
    w, h = size
    if calib and Path(calib).exists():
        cal = Calibration.load(calib)
        K, d, csize = (cal.K1, cal.d1, cal.size1) if calib_cam == 1 else (cal.K2, cal.d2, cal.size2)
        if abs(csize[0] / csize[1] - w / h) < 0.01:
            s = w / csize[0]
            K = K.copy()
            K[:2] *= s
            return K, d, f"lens from {calib} (camera {calib_cam})"
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros(5), f"lens guessed from {hfov:g} deg field of view"


class MonoSolver:
    def __init__(self, K: np.ndarray, d: np.ndarray, palm_mm: float | None = None):
        self.K, self.d, self.palm_mm = K, d, palm_mm
        self.prev: tuple[np.ndarray, np.ndarray] | None = None

    def solve(self, px: np.ndarray, world_m: np.ndarray):
        """(21,2) pixels + (21,3) metric hand model -> ((21,3) mm in camera frame, reprojection px) or None."""
        obj = world_m * 1000.0
        if self.palm_mm:
            obj = obj * (self.palm_mm / np.linalg.norm(obj[MIDDLE_MCP] - obj[WRIST]))
        if self.prev is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, px, self.K, self.d, self.prev[0].copy(), self.prev[1].copy(),
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, px, self.K, self.d, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or tvec[2, 0] <= 0:
            self.prev = None
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.d)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - px, axis=1).mean())
        if err > 25:  # lost track; start fresh next frame rather than following a bad guess
            self.prev = None
            return None
        self.prev = (rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        return obj @ R.T + tvec.ravel(), err


def run(cam: str, calib: str | None = "calib.json", calib_cam: int = 2, hfov: float = 70.0,
        palm_mm: float | None = None, udp: str | None = None, width: int = 1280, height: int = 720) -> None:
    src = Source(cam, width, height)
    K, d, lens_note = intrinsics(src.size, calib, calib_cam, hfov)
    print(lens_note)
    print(f"hand size: {f'{palm_mm:g} mm palm (measured)' if palm_mm else 'average hand assumed; pass --palm-mm for better depth'}")
    print("Keys: z engage the arm (and zero), x release it, the arm is in the browser, b set desk frame from the board lying flat, a/d/w/s rotate view, +/- zoom, q quit.")
    det = HandDetector(1)
    solver = MonoSolver(K, d, palm_mm)
    filt = OneEuro()
    Rw, tw = np.eye(3), np.zeros(3)
    desk = False
    view = View3D(desk=False)
    zero = None
    board = Detector(12.0)  # the phone board; a wrong size only shifts the origin, not the axes
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp else None
    dest = (udp.rsplit(":", 1)[0], int(udp.rsplit(":", 1)[1])) if udp else None
    arm, arm_view = ArmController(frame="camera"), Viewer()
    fps, t_prev = 0.0, time.time()
    while True:
        frame, t = src.read()
        if frame is None:
            break
        hands = det.detect_full(frame, t)
        worlds, text, payload = [], [], {"t": t, "hand": None}
        pose = None
        if hands:
            label, px, wm = hands[0]
            got = solver.solve(px, wm)
            if got is not None:
                Xc, err = got
                Xw = filt((Xc @ Rw.T + tw).ravel(), t).reshape(21, 3)
                worlds.append(Xw)
                pose = hand_pose(Xw)
                wrist = np.array(pose["wrist_mm"])
                rel = wrist - zero if zero is not None else None
                text += [
                    f"hand {label}  fit {err:.1f}px   depth {Xc[WRIST, 2]:.0f} mm from camera",
                    "wrist  x {:7.1f}  y {:7.1f}  z {:7.1f} mm".format(*wrist),
                    ("from zero  x {:7.1f}  y {:7.1f}  z {:7.1f} mm".format(*rel) if rel is not None
                     else "press z to zero"),
                    f"grip open {pose['grip_open']:.2f}   pinch {pose['pinch_mm']:.0f} mm",
                ]
                payload = {"t": t, "hand": label, "frame": "desk" if desk else "camera", "mode": "mono",
                           "reproj_px": err, "relative_mm": rel.tolist() if rel is not None else None,
                           **pose, "landmarks_mm": np.round(Xw, 1).tolist()}
        else:
            solver.prev = None
        if payload["hand"] is None:
            text.append("no hand")
        arm_state = arm.update(pose, t)
        payload["arm"] = arm_state
        if sock:
            sock.sendto(json.dumps(payload).encode(), dest)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        text.append(f"{fps:4.1f} fps   one camera   frame {'desk' if desk else 'camera'}")
        v = frame.copy()
        draw_2d(v, [(h[0], h[1]) for h in hands])
        cv2.imshow("stereo-hand mono: camera", cv2.resize(v, (800, int(v.shape[0] * 800 / v.shape[1]))))
        cv2.imshow("stereo-hand mono: 3D", view.render(worlds, text))
        arm_view.publish({**arm_state, "status": ("tracking: " + "one camera" + ("" if arm_state["engaged"] else ", press z to engage"))})
        key = cv2.waitKey(1) & 0xFF
        view.key(key)
        if key == ord("x"):
            arm.release()
        if key in (ord("q"), 27):
            break
        if key == ord("z") and worlds:
            zero = worlds[0][0].copy()
            arm.engage(pose)
        if key == ord("b"):
            bp = board_pose(board, frame, K, d)
            if bp is None:
                print("board not visible; lay it flat where the camera can see it")
            else:
                Rw, tw = desk_frame(*bp)
                arm.set_frame("desk")
                desk, view.desk, view.center, zero, filt.x = True, True, None, None, None
                view.trail.clear()
                print("desk frame set for this session: x along the board, y across it, z up")
    det.close()
    src.close()
    cv2.destroyAllWindows()
