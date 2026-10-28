"""A simulated rover arm, driven by a hand or a held phone.

The model follows the Loop rover's arm as photographed: a turret that yaws on
the chassis, a tall upper arm pushed by linear actuators, a forearm reaching
forward and down, a wrist that pitches and rolls, and a two-finger gripper.
Upper arm and forearm lengths, joint speeds and the wrist roll range come from
the team's URDF (umdloop/software-urc4). The wrist and gripper lengths, the
pitch joints' ranges and the accelerations are still estimates.

How a command becomes motion, in three stages, each there for a reason:

1. **Target.** The operator's displacement since engaging (scaled) moves a
   target point for the gripper tip, plus a wrist pitch and roll. The target
   is kept above the ground and out of the chassis, so no command can ask the
   arm to dig or hit itself.
2. **Inverse kinematics.** A damped least-squares solver finds the joint angles
   that best reach the target, starting from the previous solution and
   respecting every joint limit. Damping keeps it stable near full extension
   and over the base, where analytic solutions jump; starting from the last
   answer keeps it from flipping between elbow configurations; and when the
   target is out of reach it settles on the nearest pose it can make rather
   than failing.
3. **Motion.** Each joint chases its IK angle with its own speed and
   acceleration limit, like a geared motor or a linear actuator, so the arm
   lags a fast hand instead of teleporting, and a tracking glitch cannot
   fling it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

JOINTS = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")


@dataclass
class ArmSpec:
    shoulder_h: float = 0.14   # turret top to shoulder axis, m
    upper: float = 0.582       # shoulder to elbow (software-urc4 URDF, elbow_pitch origin)
    fore: float = 0.582        # elbow to wrist pitch axis (URDF, wrist_pitch origin)
    wrist: float = 0.13        # wrist pitch axis to gripper palm
    finger: float = 0.13       # palm to fingertips
    # Wrist roll is the URDF's +/-1.57 rad. The URDF gives the three pitch
    # joints a nominal +/-3.14 in its own zero convention, which would let this
    # model fold through itself, so those stay as estimated mechanical ranges
    # until the real stops and zero offsets are known.
    limits_deg: tuple = ((-170, 170), (-5, 120), (-165, -10), (-115, 115), (-90, 90))
    # Every arm joint in the URDF and MoveIt limits is capped at 0.5 rad/s.
    # They state no acceleration limit; these are estimates.
    max_speed_deg: tuple = (28.65, 28.65, 28.65, 28.65, 28.65)
    max_accel_deg: tuple = (120.0, 70.0, 110.0, 300.0, 400.0)
    gripper_time_s: float = 0.8   # fully open to fully closed
    scale: float = 2.5            # arm metres moved per metre of operator motion
    ground_z: float = -0.56       # wheel contact plane, relative to the turret top
    chassis: tuple = (-0.52, 0.24, -0.23, 0.23, -0.30, 0.02)  # x0 x1 y0 y1 z0 z1 keep-out box
    mast_xy: tuple = (-0.46, 0.17)
    clearance: float = 0.06


ARM = ArmSpec()
HOME_TIP = np.array([0.74, 0.0, 0.08])
HOME_PITCH = math.radians(-25)

# camera frame (x right, y down, z away from the camera, which faces you) to
# arm frame (x forward, away from you; y left; z up)
CAM_TO_ARM = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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
    side = math.cos(wr) * side0 + math.sin(wr) * up0
    return {"base": np.zeros(3), "shoulder": lift(S), "elbow": lift(E), "wrist": lift(W),
            "grip_base": lift(G), "tip": lift(T), "approach": approach, "side": side,
            "side0": side0, "yaw": yaw, "pitch": a3, "roll": wr}


def ik_seed(tip: np.ndarray, pitch: float, roll: float, spec: ArmSpec = ARM) -> np.ndarray:
    """Analytic elbow-up solution, used only to start the solver from nothing."""
    yaw = math.atan2(tip[1], tip[0])
    r = math.hypot(tip[0], tip[1])
    tool = spec.wrist + spec.finger
    wx = r - tool * math.cos(pitch)
    wz = tip[2] - tool * math.sin(pitch) - spec.shoulder_h
    d = min(max(math.hypot(wx, wz), abs(spec.upper - spec.fore) + 1e-3), spec.upper + spec.fore - 1e-3)
    c2 = (d * d - spec.upper ** 2 - spec.fore ** 2) / (2 * spec.upper * spec.fore)
    el = -math.acos(max(-1.0, min(1.0, c2)))
    sh = math.atan2(wz, wx) - math.atan2(spec.fore * math.sin(el), spec.upper + spec.fore * math.cos(el))
    q = np.array([yaw, sh, el, pitch - sh - el, roll])
    lim = np.radians(np.array(spec.limits_deg))
    return np.clip(q, lim[:, 0], lim[:, 1])


def _residual(q, tip, pitch, roll, spec, w_ang):
    k = fk(q, spec)
    return np.concatenate([k["tip"] - tip, [w_ang * _wrap(k["pitch"] - pitch), 0.5 * w_ang * _wrap(k["roll"] - roll)]])


def solve_ik(tip: np.ndarray, pitch: float, roll: float, q0: np.ndarray, spec: ArmSpec = ARM,
             iters: int = 40, w_ang: float = 0.12) -> tuple[np.ndarray, float]:
    """Damped least squares from q0, clamped to joint limits every step.
    Returns (q, position error in metres). Orientation is weighted so 1 rad
    of wrist error costs like w_ang metres of position: position wins."""
    lim = np.radians(np.array(spec.limits_deg))
    q = np.clip(np.array(q0, dtype=float), lim[:, 0], lim[:, 1])
    lam = 0.02
    r = _residual(q, tip, pitch, roll, spec, w_ang)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((5, 5))
        h = 1e-5
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = h
            J[:, j] = (_residual(q + dq, tip, pitch, roll, spec, w_ang) - r) / h
        A = J.T @ J + lam ** 2 * np.eye(5)
        step = -np.linalg.solve(A, J.T @ r)
        step = np.clip(step, -0.3, 0.3)
        qn = np.clip(q + step, lim[:, 0], lim[:, 1])
        rn = _residual(qn, tip, pitch, roll, spec, w_ang)
        cn = float(rn @ rn)
        if cn < cost:
            q, r, cost = qn, rn, cn
            lam = max(lam * 0.5, 1e-4)
            if cost < 1e-10:
                break
        else:
            lam *= 4.0
            if lam > 10:
                break
    return q, float(np.linalg.norm(fk(q, spec)["tip"] - tip))


def ik(tip: np.ndarray, pitch: float, roll: float, spec: ArmSpec = ARM, q0: np.ndarray | None = None):
    """Joint angles for a tip pose. Returns (q, reachable within 5 mm)."""
    seed = ik_seed(tip, pitch, roll, spec) if q0 is None else q0
    q, err = solve_ik(tip, pitch, roll, seed, spec)
    if q0 is not None and err > 0.005:  # warm start got stuck: try fresh
        q2, err2 = solve_ik(tip, pitch, roll, ik_seed(tip, pitch, roll, spec), spec)
        if err2 + 1e-4 < err:
            q, err = q2, err2
    return q, err < 0.005


def clamp_target(p: np.ndarray, spec: ArmSpec = ARM) -> np.ndarray:
    """Keep the commanded tip above the ground, outside the chassis and mast, and within reach."""
    p = np.array(p, dtype=float)
    p[2] = max(p[2], spec.ground_z + spec.clearance)
    x0, x1, y0, y1, z0, z1 = spec.chassis
    c = spec.clearance
    if x0 - c < p[0] < x1 + c and y0 - c < p[1] < y1 + c and p[2] < z1 + c:
        # push out through the nearest face (top, front, or a side)
        exits = {"top": z1 + c - p[2], "front": x1 + c - p[0], "left": y1 + c - p[1],
                 "right": p[1] - (y0 - c), "back": p[0] - (x0 - c)}
        face = min(exits, key=exits.get)
        if face == "top": p[2] = z1 + c
        elif face == "front": p[0] = x1 + c
        elif face == "left": p[1] = y1 + c
        elif face == "right": p[1] = y0 - c
        else: p[0] = x0 - c
    mx, my = spec.mast_xy
    d = math.hypot(p[0] - mx, p[1] - my)
    if d < 0.12:
        k = 0.12 / max(d, 1e-6)
        p[0], p[1] = mx + (p[0] - mx) * k, my + (p[1] - my) * k
    reach = spec.upper + spec.fore + spec.wrist + spec.finger - 0.01
    v = p - np.array([0, 0, spec.shoulder_h])
    n = np.linalg.norm(v)
    if n > reach:
        p = np.array([0, 0, spec.shoulder_h]) + v * reach / n
    return p


def _hand_angles(rot: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """Pitch of the pointing direction and roll about it, in the arm frame."""
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


@dataclass
class JointMotion:
    """Per-joint speed and acceleration limits: each joint brakes in time to stop on its goal."""
    vmax: np.ndarray
    amax: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def step(self, q: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        err = goal - q
        err[0], err[4] = _wrap(err[0]), _wrap(err[4])
        # fastest speed from which the joint can still stop exactly at the goal,
        # braking by amax*dt per step (the discrete-time form of sqrt(2*a*e)),
        # and never more than closes the gap in one step
        if dt <= 0:
            return q
        ad = self.amax * dt
        e = np.abs(err)
        v_brake = ad * (np.sqrt(0.25 + 2 * e / (ad * dt)) - 0.5)
        v_want = np.sign(err) * np.minimum.reduce([self.vmax, v_brake, e / dt])
        dv = np.clip(v_want - self.v, -self.amax * dt, self.amax * dt)
        self.v = self.v + dv
        nq = q + self.v * dt
        # never step past the goal (prevents chatter around it)
        over = np.sign(goal - nq) != np.sign(err)
        over &= np.abs(err) < np.abs(self.v * dt) + 1e-9
        nq[over] = q[over] + err[over]
        self.v[over] = 0.0
        return nq


class ArmController:
    def __init__(self, spec: ArmSpec = ARM, frame: str = "camera"):
        self.spec = spec
        self.M = CAM_TO_ARM if frame.startswith("camera") else np.eye(3)
        self.target = HOME_TIP.copy()
        self.pitch, self.roll, self.grip = HOME_PITCH, 0.0, 1.0
        self.q_goal, self.reachable = ik(self.target, self.pitch, self.roll, spec)
        self.q = self.q_goal.copy()
        self.g = self.grip
        self.motion = JointMotion(np.radians(spec.max_speed_deg), np.radians(spec.max_accel_deg))
        self.engaged = False
        self.ref = None
        self.t = None
        self.scale = spec.scale  # arm metres per metre of operator motion; lower for fine work

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

    def set_scale(self, scale: float, pose: dict | None = None) -> None:
        """Change how far the arm moves per unit of operator motion. Re-references
        the clutch at the current pose, so switching mid-move never jumps the target."""
        if abs(scale - self.scale) < 1e-9:
            return
        self.scale = scale
        if self.engaged and pose is not None:
            self.engage(pose)

    def go_home(self) -> None:
        """Let go and send the arm back to its rest pose, at its normal joint speeds."""
        self.release()
        self.command(HOME_TIP, HOME_PITCH, 0.0)

    def at_limit(self, margin_deg: float = 1.0) -> list[str]:
        lim = np.radians(np.array(self.spec.limits_deg))
        m = math.radians(margin_deg)
        return [n for n, v, (lo, hi) in zip(JOINTS[:4], self.q_goal, lim) if v <= lo + m or v >= hi - m]

    def command(self, tip: np.ndarray, pitch: float, roll: float, grip: float | None = None) -> None:
        """Set the target directly (used by the scripted demo)."""
        self.target = clamp_target(tip, self.spec)
        self.pitch = float(np.clip(pitch, -1.6, 1.2))
        self.roll = _wrap(roll)
        if grip is not None:
            self.grip = float(np.clip(grip, 0, 1))

    def update(self, pose: dict | None, t: float) -> dict:
        if self.engaged and pose is not None:
            hand0, p0, r0, tip0, pitch0, roll0 = self.ref
            hand = self.M @ np.array(pose["wrist_mm"]) / 1000.0
            p, r = _hand_angles(pose["rotation"], self.M)
            self.command(tip0 + self.scale * (hand - hand0), pitch0 + _wrap(p - p0),
                         roll0 + _wrap(r - r0), pose["grip_open"])
        self.q_goal, self.reachable = ik(self.target, self.pitch, self.roll, self.spec, q0=self.q_goal)
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 0.1)
        self.t = t
        self.q = self.motion.step(self.q, self.q_goal, dt)
        rate = dt / self.spec.gripper_time_s
        self.g += float(np.clip(self.grip - self.g, -rate, rate))
        return self.state()

    def state(self) -> dict:
        k = fk(self.q, self.spec)
        return {
            "engaged": self.engaged, "reachable": bool(self.reachable), "gripper_open": round(self.g, 3),
            "q": [round(float(v), 5) for v in self.q],
            "joints_deg": {n: round(math.degrees(v), 1) for n, v in zip(JOINTS, self.q)},
            "speed_frac": [round(float(abs(v) / m), 3) for v, m in zip(self.motion.v, self.motion.vmax)],
            "target_tip_m": np.round(self.target, 4).tolist(),
            "tip_m": np.round(k["tip"], 4).tolist(),
            "lag_m": round(float(np.linalg.norm(k["tip"] - self.target)), 4),
            "scale": self.scale, "at_limit": self.at_limit(),
        }


def spec_dict(spec: ArmSpec = ARM) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.__dict__.items()}
