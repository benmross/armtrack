"""A handheld wand: a printed cube of ArUco markers, one webcam, and a gyro.

The bare-hand trackers in this package estimate depth from how big a hand
looks (mono) or need two calibrated cameras (track). A marker of known printed
size removes that guess: one camera recovers full position and rotation,
because the marker's true size fixes the distance.

The wand is a 3-D printed grip with a cube on top. Five faces carry a 4x4
ArUco marker each (the bottom face is where the grip attaches), so from any
direction the operator can comfortably hold it at least one face, and usually
two or three, face the camera. All visible corners go into one solvePnP, which
is far steadier than any single marker: corners on faces at right angles to
each other pin rotation down in every axis.

An MPU-6050 gyro in the grip covers what a camera handles badly:

- **One face visible.** A single flat marker has two rotations that fit it
  almost equally well (the planar ambiguity), and solvers flip between them.
  The gyro's prediction picks the right one.
- **Motion blur and brief occlusion.** Between camera fixes, orientation runs
  on the gyro at 200 Hz. The camera then corrects the gyro's drift and learns
  its bias, so neither has to be good alone.

Position comes from the camera only. Integrating an accelerometer twice drifts
by centimetres within a second, which is why no IMU-only design can hold
position. When the camera loses the wand for longer than HOLD_S, tracking is
reported lost and the arm controller lets go, the same deadman the phone uses.

Wand frame: origin at the cube's centre, x to the operator's right, y forward
(where the wand points), z up toward the top face.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
FACES = {  # marker id -> outward normal, in the wand frame
    0: (0, 0, 1),    # top
    1: (0, 1, 0),    # front, facing away from the operator
    2: (1, 0, 0),    # right
    3: (0, -1, 0),   # back, facing the operator
    4: (-1, 0, 0),   # left
}
HOLD_S = 0.25       # how long position may be held without a camera fix
MAX_REPROJ_PX = 4.0
TO_HAND = np.diag([-1.0, 1.0, -1.0])  # wand axes -> the hand axes ArmController expects (y forward, z out of the palm)


@dataclass
class Cube:
    side_mm: float = 60.0     # printed cube edge
    marker_mm: float = 50.0   # black square edge, as measured on the printout

    def face_axes(self, n):
        """(u, v) for a face: image right and image down when looking at it from outside."""
        n = np.asarray(n, float)
        v = np.array([0.0, -1.0, 0.0]) if abs(n[2]) > 0.5 else np.array([0.0, 0.0, -1.0])
        u = np.cross(v, -n)
        return u, v

    def corners(self, mid: int, size_mm: float | None = None) -> np.ndarray:
        """(4,3) corners in mm, in ArUco order (top-left, top-right, bottom-right, bottom-left)."""
        n = np.asarray(FACES[mid], float)
        u, v = self.face_axes(n)
        c = n * self.side_mm / 2
        h = (size_mm or self.marker_mm) / 2
        return np.array([c - h * u - h * v, c + h * u - h * v, c + h * u + h * v, c - h * u + h * v])


def _rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.acos(max(-1.0, min(1.0, c)))


def _exp(w: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))[0]


def _log(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0].ravel()


class CubeSolver:
    """Image -> wand pose in the camera frame."""

    def __init__(self, K: np.ndarray, d: np.ndarray, cube: Cube = Cube()):
        self.K, self.d, self.cube = K, np.asarray(d, float), cube
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)

    def detect(self, gray: np.ndarray):
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners) if int(i) in FACES]

    def solve(self, gray: np.ndarray, predict: np.ndarray | None = None):
        """Return (R, t_mm, faces_seen, reproj_px) or None.

        `predict` is the expected rotation (from the gyro or the last frame); it
        picks between the two fits of a lone marker and seeds the solver."""
        seen = self.detect(gray)
        if not seen:
            return None
        obj = np.concatenate([self.cube.corners(i) for i, _ in seen]).astype(np.float64)
        img = np.concatenate([c for _, c in seen]).astype(np.float64)
        if len(seen) == 1:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.K, self.d, flags=cv2.SOLVEPNP_IPPE)
            cands = [(cv2.Rodrigues(r)[0], t, float(e)) for r, t, e in zip(rvecs, tvecs, np.ravel(errs))]
            if predict is not None:
                R, t, _ = min(cands, key=lambda c: _rotation_angle(c[0], predict))
            else:
                R, t, _ = min(cands, key=lambda c: c[2])
            rvec = cv2.Rodrigues(R)[0]
        else:
            ok, rvec, t = cv2.solvePnP(obj, img, self.K, self.d, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None
        rvec, t = cv2.solvePnPRefineLM(obj, img, self.K, self.d, rvec, t)
        proj = cv2.projectPoints(obj, rvec, t, self.K, self.d)[0].reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        if err > MAX_REPROJ_PX:
            return None
        return cv2.Rodrigues(rvec)[0], t.ravel(), len(seen), err


class OrientationFilter:
    """Gyro integration corrected by the camera: a complementary filter on SO(3)
    that also learns the gyro's bias, so orientation survives camera dropouts."""

    def __init__(self, k: float = 0.15, ki: float = 0.02):
        self.R: np.ndarray | None = None   # wand -> camera
        self.bias = np.zeros(3)
        self.k, self.ki = k, ki

    def gyro(self, w_body: np.ndarray, dt: float) -> None:
        if self.R is not None and 0 < dt < 0.1:
            self.R = self.R @ _exp((np.asarray(w_body) - self.bias) * dt)

    def vision(self, R_meas: np.ndarray) -> None:
        if self.R is None or _rotation_angle(self.R, R_meas) > math.radians(30):
            self.R = R_meas.copy()  # first fix, or the estimate is lost: trust the camera
            return
        e = _log(self.R.T @ R_meas)  # body-frame error
        self.R = self.R @ _exp(self.k * e)
        self.bias -= self.ki * e


class WandTracker:
    """Camera fixes plus gyro, out as the pose dict ArmController takes."""

    def __init__(self, solver: CubeSolver, use_gyro: bool = True):
        self.solver, self.use_gyro = solver, use_gyro
        self.ori = OrientationFilter()
        self.pos: np.ndarray | None = None
        self.t_fix = -1e9
        self.last = None

    def gyro(self, w_body: np.ndarray, dt: float) -> None:
        if self.use_gyro:
            self.ori.gyro(w_body, dt)

    def frame(self, gray: np.ndarray, t: float):
        got = self.solver.solve(gray, self.ori.R)
        if got is not None:
            R, tmm, faces, err = got
            self.ori.vision(R) if self.use_gyro else setattr(self.ori, "R", R)
            self.pos, self.t_fix = tmm, t
            self.last = {"faces": faces, "reproj_px": err}
        return got

    def pose(self, t: float, grip_open: float = 1.0) -> dict | None:
        if self.pos is None or self.ori.R is None or t - self.t_fix > HOLD_S:
            return None
        return {"wrist_mm": self.pos.tolist(), "rotation": (self.ori.R @ TO_HAND).tolist(),
                "grip_open": grip_open, "age_s": round(t - self.t_fix, 3)}


# ---- the grip's microcontroller --------------------------------------------

PACKET = struct.Struct("<2s6hHBIB")  # sync, gyro xyz, accel xyz, trigger, buttons, micros, checksum
SYNC = b"\xa5\x5a"
GYRO_LSB_PER_DPS = 65.5   # MPU-6050 at +/-500 deg/s
ACCEL_LSB_PER_G = 8192.0  # +/-4 g


def parse_packets(buf: bytearray) -> list[dict]:
    """Pull every complete, checksummed packet off the front of `buf` (consumed in place)."""
    out = []
    while True:
        i = buf.find(SYNC)
        if i < 0:
            del buf[:max(0, len(buf) - 1)]
            return out
        del buf[:i]
        if len(buf) < PACKET.size:
            return out
        raw = bytes(buf[:PACKET.size])
        if sum(raw[2:-1]) & 0xFF != raw[-1]:
            del buf[:1]  # false sync or a corrupted packet: resync
            continue
        del buf[:PACKET.size]
        _, gx, gy, gz, ax, ay, az, trig, btn, us, _ = PACKET.unpack(raw)
        out.append({
            "gyro": np.radians(np.array([gx, gy, gz]) / GYRO_LSB_PER_DPS),
            "accel_g": np.array([ax, ay, az]) / ACCEL_LSB_PER_G,
            "trigger": trig / 1023.0, "clutch": bool(btn & 1), "home": bool(btn & 2),
            "fine": bool(btn & 4), "us": us,
        })


class GripLink:
    """Reads the grip's serial stream on a thread; the gyro goes straight into the tracker."""

    def __init__(self, port: str, tracker: WandTracker, imu_to_wand: np.ndarray | None = None, baud: int = 115200):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.tracker = tracker
        self.M = np.eye(3) if imu_to_wand is None else imu_to_wand
        self.lock = threading.Lock()
        self.state = {"trigger": 0.0, "clutch": False, "home": False, "fine": False}
        self.t_rx = 0.0
        self.us_prev = None
        self.still: list[np.ndarray] = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        buf = bytearray()
        while True:
            buf += self.ser.read(256)
            for p in parse_packets(buf):
                w = self.M @ p["gyro"]
                dt = 0.0 if self.us_prev is None else ((p["us"] - self.us_prev) & 0xFFFFFFFF) / 1e6
                self.us_prev = p["us"]
                with self.lock:
                    if len(self.still) < 200:  # first second: assume the wand is set down, take the bias
                        self.still.append(w)
                        if len(self.still) == 200:
                            self.tracker.ori.bias = np.mean(self.still, axis=0)
                    else:
                        self.tracker.gyro(w, dt)
                    self.state = {k: p[k] for k in ("trigger", "clutch", "home", "fine")}
                    self.t_rx = time.time()


# ---- printables --------------------------------------------------------------

def render_markers(out: str, cube: Cube = Cube(), dpi: int = 300, cut_mm: float | None = None) -> None:
    """A letter page with the five face markers at exact size, each in a square to cut out.

    The square is 4 mm under the face so it drops into the printed cube's recess;
    pass cut_mm equal to the side for a homemade cube, whose edges then line the
    paper up. Writes a PDF (exact scale when printed at 100%) or a PNG."""
    px = lambda mm: int(round(mm / 25.4 * dpi))
    page = np.full((px(279.4), px(215.9)), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    face, mark = px(cut_mm or cube.side_mm - 4), px(cube.marker_mm)
    names = ["TOP", "FRONT", "RIGHT", "BACK", "LEFT"]
    for k, mid in enumerate(FACES):
        r, c = divmod(k, 3)
        y0, x0 = px(24) + r * (face + px(22)), px(12) + c * (face + px(8))
        tile = np.full((face, face), 255, np.uint8)
        o = (face - mark) // 2
        tile[o:o + mark, o:o + mark] = cv2.aruco.generateImageMarker(d, mid, mark, borderBits=1)
        cv2.rectangle(tile, (0, 0), (face - 1, face - 1), 150, 2)
        page[y0:y0 + face, x0:x0 + face] = tile
        # which way up the square goes: the bottom edge of every side face is toward the handle,
        # and the top face's bottom edge is toward the back face (the operator)
        edge = "toward BACK face" if mid == 0 else "toward HANDLE"
        cv2.putText(page, f"{names[mid]}  (id {mid})", (x0, y0 + face + px(6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.putText(page, f"this edge {edge}", (x0, y0 + face + px(12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.putText(page, f"Print at 100% (no fit to page). Each black square must measure {cube.marker_mm:g} mm.",
                (px(12), px(10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.putText(page, "Cut on the grey lines. Holding the handle, FRONT faces the way you point and RIGHT is on your right.",
                (px(12), px(16)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    y = px(265)
    cv2.line(page, (px(12), y), (px(112), y), 0, 3)  # a 100 mm scale check
    for mm in range(0, 101, 10):
        cv2.line(page, (px(12 + mm), y - px(3 if mm % 50 else 5)), (px(12 + mm), y), 0, 2)
    cv2.putText(page, "100 mm", (px(116), y + px(1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    if out.lower().endswith(".pdf"):
        from PIL import Image
        Image.fromarray(page).save(out, resolution=dpi)
    else:
        cv2.imwrite(out, page, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# ---- live ---------------------------------------------------------------------

def run(cam: str, port: str | None = None, calib: str | None = "calib.json", calib_cam: int = 2,
        hfov: float = 70.0, udp: str | None = None, width: int = 1280, height: int = 720,
        side_mm: float = 60.0, marker_mm: float = 50.0, no_gyro: bool = False,
        grip_mode: str = "rate") -> None:
    import json
    import socket

    from .arm import ArmController
    from .cameras import Source
    from .mono import intrinsics
    from .phone import heading_frame
    from .viewer import Viewer

    src = Source(cam, width, height)
    K, d, note = intrinsics(src.size, calib, calib_cam, hfov)
    print(note)
    tracker = WandTracker(CubeSolver(K, d, Cube(side_mm, marker_mm)), use_gyro=not no_gyro and port is not None)
    grip = GripLink(port, tracker) if port else None
    if grip:
        print("keep the wand still for one second while the gyro bias is measured")
    print("Hold the clutch (or z/x on the keyboard without a grip) to drive the arm; q quits.")
    arm, view = ArmController(frame="camera"), Viewer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp else None
    dest = (udp.rsplit(":", 1)[0], int(udp.rsplit(":", 1)[1])) if udp else None
    key_clutch, engaged_prev, home_prev = False, False, False
    grip_open, t_prev = 1.0, time.time()
    while True:
        frame, t = src.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with grip.lock if grip else threading.Lock():
            got = tracker.frame(gray, t)
            buttons = dict(grip.state) if grip else {"trigger": 0.0, "clutch": key_clutch, "home": False, "fine": False}
        if grip_mode == "rate":  # spring-centred joystick: push up to open, down to close
            push = 2 * buttons["trigger"] - 1
            if abs(push) > 0.15:
                grip_open = float(np.clip(grip_open + push * 1.2 * (t - t_prev), 0, 1))
        else:
            grip_open = 1.0 - buttons["trigger"]
        t_prev = t
        pose = tracker.pose(t, grip_open=grip_open)
        if grip and time.time() - grip.t_rx > HOLD_S:
            buttons["clutch"] = False  # the grip's cable came out: let go
        engaged = buttons["clutch"] and pose is not None
        if buttons["home"] and not home_prev:
            arm.go_home()
            engaged = False
        home_prev = buttons["home"]
        if engaged:
            arm.set_scale(0.6 if buttons["fine"] else arm.spec.scale, pose)
        if engaged and not engaged_prev:
            arm.M = heading_frame({"rotation": (CAM_Y_UP @ np.array(pose["rotation"])).tolist()}, arm.M) @ CAM_Y_UP
            arm.engage(pose)
        elif not engaged and engaged_prev:
            arm.release()
        engaged_prev = engaged
        if pose is not None:
            arm.grip = pose["grip_open"]
        st = arm.update(pose if engaged else None, t)
        if sock:
            sock.sendto(json.dumps({"t": t, "mode": "wand", "pose": pose, "arm": st}).encode(), dest)
        view.publish({**st, "status": "wand: " + ("lost" if pose is None else
                      f"{tracker.last['faces']} faces, {tracker.last['reproj_px']:.1f}px")})
        vis = frame.copy()
        if got:
            cv2.drawFrameAxes(vis, K, d, cv2.Rodrigues(got[0])[0], got[1], side_mm)
        cv2.imshow("wand", cv2.resize(vis, (800, int(vis.shape[0] * 800 / vis.shape[1]))))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("z"):
            key_clutch = True
        if k == ord("x"):
            key_clutch = False
    src.close()
    cv2.destroyAllWindows()


# camera frame (y down, z forward) -> a y-up frame like ARKit's, so heading_frame can be reused
CAM_Y_UP = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
