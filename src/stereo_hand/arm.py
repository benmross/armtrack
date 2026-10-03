"""A simulated rover arm driven by the tracked hand.

The model follows the Loop rover's arm as photographed: a turret that yaws on
the chassis, a tall upper arm pushed by a linear actuator, a forearm reaching
forward and down, a wrist that pitches and rolls, and a two-finger gripper.
Link lengths are estimates from a photo, not the team's CAD; change ARM below
when real numbers exist.

Control is a clutch, the way teleop usually works: press z to engage, and from
then on the gripper tip moves by your hand's displacement (scaled), wrist pitch
and roll follow your hand's, and the gripper opens as your hand opens. Press x
to let go; the arm holds still. Engaging again carries on from where the arm is,
so you can re-centre your hand without moving the arm. Joints are rate-limited
like real motors, which is also why a tracking glitch cannot fling the arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArmSpec:
    shoulder_h: float = 0.12   # turret top to shoulder axis, m
    upper: float = 0.50        # shoulder to elbow
    fore: float = 0.55         # elbow to wrist pitch axis
    wrist: float = 0.13        # wrist pitch axis to gripper base
    finger: float = 0.15
    limits_deg: tuple = ((-170, 170), (-10, 125), (-170, 0), (-115, 115), (-180, 180))
    max_speed_deg: float = 120.0  # per joint, per second
    scale: float = 2.5           # arm metres moved per metre of hand motion


ARM = ArmSpec()
HOME_TIP = np.array([0.72, 0.0, 0.10])
HOME_PITCH = math.radians(-25)

# camera frame (x right, y down, z away from the camera, which faces you) to
# arm frame (x forward, away from you; y left; z up)
CAM_TO_ARM = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def fk(q: np.ndarray, spec: ArmSpec = ARM) -> dict:
    """Joint angles (rad: yaw, shoulder, elbow, wrist pitch, wrist roll) -> key points in metres."""
    yaw, sh, el, wp, wr = q
    a1, a2 = sh, sh + el
    a3 = a2 + wp
    S = np.array([0.0, spec.shoulder_h])
    E = S + spec.upper * np.array([math.cos(a1), math.sin(a1)])
    W = E + spec.fore * np.array([math.cos(a2), math.sin(a2)])
    G = W + spec.wrist * np.array([math.cos(a3), math.sin(a3)])
    T = G + spec.finger * np.array([math.cos(a3), math.sin(a3)])
    c, s = math.cos(yaw), math.sin(yaw)

    def lift(p):
        return np.array([p[0] * c, p[0] * s, p[1]])

    approach = np.array([math.cos(a3) * c, math.cos(a3) * s, math.sin(a3)])
    side0 = np.array([-s, c, 0.0])
    up0 = np.cross(approach, side0)
    side = math.cos(wr) * side0 + math.sin(wr) * up0  # gripper opening direction after roll
    return {"base": np.zeros(3), "shoulder": lift(S), "elbow": lift(E), "wrist": lift(W),
            "grip_base": lift(G), "tip": lift(T), "approach": approach, "side": side,
            "side0": side0, "yaw": yaw}


def ik(tip: np.ndarray, pitch: float, roll: float, spec: ArmSpec = ARM) -> tuple[np.ndarray, bool]:
    """Gripper tip position and tool pitch/roll -> joint angles (elbow up). Returns (q, reachable)."""
    yaw = math.atan2(tip[1], tip[0])
    r = math.hypot(tip[0], tip[1])
    tool = spec.wrist + spec.finger
    wx = r - tool * math.cos(pitch)
    wz = tip[2] - tool * math.sin(pitch) - spec.shoulder_h
    d = math.hypot(wx, wz)
    lo, hi = abs(spec.upper - spec.fore) + 1e-3, spec.upper + spec.fore - 1e-3
    reachable = lo <= d <= hi
    if not reachable:  # reach as far as possible in the same direction
        k = min(max(d, lo), hi) / max(d, 1e-9)
        wx, wz, d = wx * k, wz * k, min(max(d, lo), hi)
    c2 = (d * d - spec.upper ** 2 - spec.fore ** 2) / (2 * spec.upper * spec.fore)
    el = -math.acos(max(-1.0, min(1.0, c2)))
    sh = math.atan2(wz, wx) - math.atan2(spec.fore * math.sin(el), spec.upper + spec.fore * math.cos(el))
    wp = pitch - (sh + el)
    q = np.array([yaw, sh, el, wp, roll])
    lim = np.radians(np.array(spec.limits_deg))
    clipped = np.clip(q, lim[:, 0], lim[:, 1])
    return clipped, reachable and np.allclose(q, clipped)


def _hand_angles(rot: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """Pitch of the fingers' direction and roll of the palm about it, in the arm frame."""
    R = M @ np.asarray(rot)
    fwd, palm_out = R[:, 1], R[:, 2]
    pitch = math.atan2(fwd[2], math.hypot(fwd[0], fwd[1]))
    side = np.cross(fwd, [0, 0, 1.0])
    if np.linalg.norm(side) < 1e-6:
        return pitch, 0.0
    side /= np.linalg.norm(side)
    up = np.cross(side, fwd)
    roll = math.atan2(float(palm_out @ side), float(-palm_out @ up))
    return pitch, roll


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ArmController:
    def __init__(self, spec: ArmSpec = ARM, frame: str = "camera"):
        self.spec = spec
        self.M = CAM_TO_ARM if frame.startswith("camera") else np.eye(3)
        self.target = HOME_TIP.copy()
        self.pitch, self.roll, self.grip = HOME_PITCH, 0.0, 0.5
        self.q, _ = ik(self.target, self.pitch, self.roll, spec)
        self.g = self.grip
        self.engaged = False
        self.reachable = True
        self.ref = None
        self.t = None

    def set_frame(self, frame: str) -> None:
        self.M = CAM_TO_ARM if frame.startswith("camera") else np.eye(3)
        self.release()

    def engage(self, pose: dict) -> None:
        p, r = _hand_angles(pose["rotation"], self.M)
        self.ref = (self.M @ np.array(pose["wrist_mm"]) / 1000.0, p, r,
                    self.target.copy(), self.pitch, self.roll)
        self.engaged = True

    def release(self) -> None:
        self.engaged = False
        self.ref = None

    def update(self, pose: dict | None, t: float) -> dict:
        if self.engaged and pose is not None:
            hand0, p0, r0, tip0, pitch0, roll0 = self.ref
            hand = self.M @ np.array(pose["wrist_mm"]) / 1000.0
            p, r = _hand_angles(pose["rotation"], self.M)
            self.target = tip0 + self.spec.scale * (hand - hand0)
            self.pitch = pitch0 + _wrap(p - p0)
            self.roll = _wrap(roll0 + _wrap(r - r0))
            self.grip = pose["grip_open"]
        goal, self.reachable = ik(self.target, self.pitch, self.roll, self.spec)
        dt = 0.0 if self.t is None else min(t - self.t, 0.2)
        self.t = t
        step = math.radians(self.spec.max_speed_deg) * dt
        dq = goal - self.q
        dq[0], dq[4] = _wrap(dq[0]), _wrap(dq[4])
        self.q = self.q + np.clip(dq, -step, step)
        self.g += float(np.clip(self.grip - self.g, -2.0 * dt, 2.0 * dt))
        return self.state()

    def state(self) -> dict:
        names = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
        return {"engaged": self.engaged, "reachable": self.reachable, "gripper_open": round(self.g, 3),
                "joints_deg": {n: round(math.degrees(v), 1) for n, v in zip(names, self.q)},
                "target_tip_m": np.round(self.target, 3).tolist()}


# --- drawing -----------------------------------------------------------------

SILVER, SILVER_DK = (205, 200, 196), (120, 116, 112)
BLACK, BLACK_HI = (28, 28, 30), (70, 70, 74)
WHITE, WHITE_DK = (238, 238, 240), (150, 150, 155)
RED, RED_DK = (52, 40, 225), (30, 20, 140)
CHASSIS = (185, 182, 178)


class ArmView:
    def __init__(self, size: int = 640):
        self.size = size
        self.yaw, self.pitch, self.scale = -18.0, 12.0, 300.0
        self.center = np.array([0.2, 0.0, 0.05])

    def key(self, k: int) -> None:
        if k == ord("j"): self.yaw -= 8
        elif k == ord("l"): self.yaw += 8
        elif k == ord("i"): self.pitch = min(85, self.pitch + 6)
        elif k == ord("k"): self.pitch = max(-30, self.pitch - 6)

    def _basis(self):
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        right = np.array([cy, sy, 0.0])
        fwd = np.array([-sy, cy, 0.0])
        up = cp * np.array([0, 0, 1.0]) - sp * fwd
        depth = cp * fwd + sp * np.array([0, 0, 1.0])
        return right, up, depth

    def _p(self, X):
        right, up, depth = self._basis()
        Y = np.atleast_2d(X) - self.center
        px = np.stack([self.size / 2 + self.scale * (Y @ right), self.size * 0.55 - self.scale * (Y @ up)], -1)
        return px, Y @ depth

    def render(self, q: np.ndarray, g: float, target: np.ndarray | None, text: list[str]) -> np.ndarray:
        img = np.full((self.size, self.size, 3), (34, 30, 28), np.uint8)
        _, _, view_dir = self._basis()
        prims = []  # (depth, kind, data); drawn far to near

        def tube(a, b, radius, col, dark, behind=False):
            d = float(self._p((a + b) / 2)[1][0]) + (100.0 if behind else 0.0)
            prims.append((d, "tube", (a, b, radius, col, dark)))

        def poly(pts, col):
            prims.append((float(self._p(np.mean(pts, 0))[1][0]), "poly", (np.array(pts), col)))

        def wheel(hub, axis, radius, half_w):
            prims.append((float(self._p(hub)[1][0]), "wheel", (hub, axis, radius, half_w)))

        def disc(c, n, radius, col, k=20):
            n = n / np.linalg.norm(n)
            u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
            poly(c + radius * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)), col)

        # floor grid
        floor = -0.56
        for v in np.arange(-1.0, 1.21, 0.2):
            for a, b in (([v, -1.0], [v, 1.0]), ([-1.0, v], [1.2, v])):
                p, _ = self._p(np.array([[a[0], a[1], floor], [b[0], b[1], floor]]))
                cv2.line(img, tuple(map(int, p[0])), tuple(map(int, p[1])), (52, 48, 46), 1)

        # chassis: a riveted aluminium box, shaded by which way each face points
        x0, x1, y0, y1, z0, z1 = -0.50, 0.22, -0.20, 0.20, -0.22, 0.0
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        faces = [((0, 1, 3, 2), [-1, 0, 0]), ((4, 5, 7, 6), [1, 0, 0]), ((0, 1, 5, 4), [0, -1, 0]),
                 ((2, 3, 7, 6), [0, 1, 0]), ((1, 3, 7, 5), [0, 0, 1]), ((0, 2, 6, 4), [0, 0, -1])]
        light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
        for idx, n in faces:
            if np.dot(n, view_dir) > 0:  # facing away from the viewer
                continue
            shade = 0.55 + 0.45 * max(0.0, float(np.dot(n, light)))
            poly(corners[list(idx)], tuple(int(c * shade) for c in CHASSIS))

        # rocker legs and six red wheels
        for wx in (-0.62, -0.14, 0.34):
            for side in (-1, 1):
                hub = np.array([wx, side * 0.40, floor + 0.12])
                tube(np.array([np.clip(wx, x0, x1), side * 0.18, -0.15]), hub + [0, -side * 0.05, 0.03],
                     0.018, SILVER, SILVER_DK, behind=True)  # legs sit under the box and behind the tyres
                wheel(hub, np.array([0, 1.0, 0]), 0.12, 0.05)

        k = fk(q)
        yaw = k["yaw"]
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        # turret
        tube(np.array([0, 0, 0.0]), np.array([0, 0, 0.07]), 0.075, BLACK_HI, BLACK)
        tube(k["shoulder"] - 0.09 * k["side0"], k["shoulder"] + 0.09 * k["side0"], 0.045, WHITE, WHITE_DK)
        # upper arm and its black linear actuator, anchored behind the shoulder
        tube(k["shoulder"], k["elbow"], 0.032, SILVER, SILVER_DK)
        anchor = np.array([0, 0, 0.06]) - 0.16 * radial + 0.05 * k["side0"]
        on_arm = k["shoulder"] + 0.62 * (k["elbow"] - k["shoulder"]) + 0.05 * k["side0"]
        tube(anchor, anchor + 0.55 * (on_arm - anchor), 0.026, BLACK, BLACK)
        tube(anchor + 0.5 * (on_arm - anchor), on_arm, 0.014, BLACK_HI, BLACK)
        # elbow bracket and forearm
        tube(k["elbow"] - 0.06 * k["side0"], k["elbow"] + 0.06 * k["side0"], 0.05, BLACK_HI, BLACK)
        tube(k["elbow"], k["wrist"], 0.028, SILVER, SILVER_DK)
        # wrist: white housing with a black pitch joint
        tube(k["wrist"] - 0.05 * k["side0"], k["wrist"] + 0.05 * k["side0"], 0.05, BLACK_HI, BLACK)
        tube(k["wrist"], k["grip_base"], 0.048, WHITE, WHITE_DK)
        tube(k["grip_base"] - 0.04 * k["side"], k["grip_base"] + 0.04 * k["side"], 0.03, BLACK_HI, BLACK)
        # gripper: two red fingers spread along the rolled side axis
        spread = 0.012 + 0.045 * g
        for s in (-1, 1):
            base = k["grip_base"] + s * spread * k["side"]
            tube(base, base + ARM.finger * k["approach"] - s * 0.008 * k["side"], 0.02, RED, RED_DK)

        prims.sort(key=lambda p: -p[0])
        for _, kind, data in prims:
            if kind == "wheel":
                self._draw_wheel(img, *data)
                continue
            if kind == "poly":
                pts, col = data
                p, _ = self._p(pts)
                cv2.fillPoly(img, [p.astype(np.int32)], col, cv2.LINE_AA)
                cv2.polylines(img, [p.astype(np.int32)], True, tuple(int(c * 0.6) for c in col), 1, cv2.LINE_AA)
            else:
                a, b, radius, col, dark = data
                p, _ = self._p(np.array([a, b]))
                w = max(2, int(2 * radius * self.scale))
                pa, pb = tuple(map(int, p[0])), tuple(map(int, p[1]))
                cv2.line(img, pa, pb, dark, w + 2, cv2.LINE_AA)
                cv2.line(img, pa, pb, col, max(1, w - 2), cv2.LINE_AA)
                hi = tuple(min(255, int(c * 1.12) + 12) for c in col)
                off = np.array([0, -max(1, w // 4)])
                cv2.line(img, tuple(map(int, p[0] + off)), tuple(map(int, p[1] + off)), hi, max(1, w // 5), cv2.LINE_AA)

        if target is not None:
            p, _ = self._p(target)
            c = tuple(map(int, p[0]))
            cv2.drawMarker(img, c, (90, 220, 120), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        for i, line in enumerate(text):
            cv2.putText(img, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img


def _ring(c, n, radius, k=28):
    u = np.cross(n, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return c + radius * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)), ang


def _draw_wheel(self, img, hub, axis, radius, half_w):
    """A real cylinder: the hull of both rims for the tread, then the cap that faces the viewer."""
    _, _, view_dir = self._basis()
    a = axis / np.linalg.norm(axis)
    near = -a if np.dot(a, view_dir) > 0 else a  # the cap pointing toward the viewer
    rim_n, ang = _ring(hub + near * half_w, a, radius)
    rim_f, _ = _ring(hub - near * half_w, a, radius)
    pn, _ = self._p(rim_n)
    pf, _ = self._p(rim_f)
    hull = cv2.convexHull(np.vstack([pn, pf]).astype(np.int32))
    cv2.fillPoly(img, [hull], RED_DK, cv2.LINE_AA)
    for i in range(0, len(ang), 2):  # tread grooves across the tyre
        cv2.line(img, tuple(map(int, pn[i])), tuple(map(int, pf[i])), (24, 16, 110), 1, cv2.LINE_AA)
    cv2.fillPoly(img, [pn.astype(np.int32)], RED, cv2.LINE_AA)
    cv2.polylines(img, [pn.astype(np.int32)], True, RED_DK, 2, cv2.LINE_AA)
    hub_ring, _ = _ring(hub + near * (half_w + 0.002), a, radius * 0.5)
    ph, _ = self._p(hub_ring)
    cv2.fillPoly(img, [ph.astype(np.int32)], SILVER_DK, cv2.LINE_AA)
    cv2.polylines(img, [ph.astype(np.int32)], True, (80, 78, 76), 1, cv2.LINE_AA)


ArmView._draw_wheel = _draw_wheel


def arm_text(st: dict) -> list[str]:
    j = st["joints_deg"]
    head = ("ENGAGED: arm follows your hand   (x releases)" if st["engaged"]
            else "press z to engage the arm")
    lines = [head,
             f"yaw {j['base_yaw']:6.1f}  shoulder {j['shoulder']:6.1f}  elbow {j['elbow']:6.1f}",
             f"wrist pitch {j['wrist_pitch']:6.1f}  roll {j['wrist_roll']:6.1f}  grip {st['gripper_open']:.2f}"]
    if not st["reachable"]:
        lines.append("target out of reach: holding at the limit")
    lines.append("j/l spin, i/k tilt this view")
    return lines
