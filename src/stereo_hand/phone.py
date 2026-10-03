"""Drive the rover arm from an iPhone running Arm Pilot (ios/ArmPilot).

The phone tracks its own position and rotation with ARKit, which fuses the
camera with the motion sensors, so position does not drift the way integrating
the accelerometer would, and has none of the depth jitter of guessing distance
from a hand's apparent size. It sends one packet per camera frame (60 Hz).

Holding a thumb on the phone engages the arm (a clutch); lifting it lets go and
the arm holds. "Forward" is wherever the phone's top edge points at the moment
you engage, so you can stand anywhere and face any way.
"""

from __future__ import annotations

import json
import math
import socket
import time

import cv2
import numpy as np

from .arm import ArmController, ArmView, arm_text


def phone_pose(msg: dict) -> dict:
    """ARKit camera pose -> the pose dict the arm controller takes for a hand.

    ARKit's camera axes: x toward the phone's bottom edge (the sensor is
    landscape-native), y toward its right side, z out of the screen. Held
    screen-up like a remote, the top edge (-x) plays the fingers and the back
    of the phone (-z) plays the palm, which faces down."""
    rot = np.array(msg["rot"], dtype=float)  # columns: camera x, y, z in world (y up)
    xc, yc, zc = rot[:, 0], rot[:, 1], rot[:, 2]
    hand_R = np.column_stack([-yc, -xc, -zc])
    return {"wrist_mm": (np.array(msg["pos"]) * 1000.0).tolist(), "rotation": hand_R.tolist(),
            "grip_open": 0.0 if msg.get("grip") else 1.0}


def heading_frame(pose: dict, fallback: np.ndarray) -> np.ndarray:
    """World -> arm rotation with arm x along the phone's current pointing direction, flattened."""
    fwd = np.array(pose["rotation"])[:, 1].copy()
    fwd[1] = 0.0  # ARKit y is up
    n = np.linalg.norm(fwd)
    if n < 0.2:  # pointing nearly straight up or down: heading is meaningless
        return fallback
    fwd /= n
    up = np.array([0.0, 1.0, 0.0])
    left = np.cross(up, fwd)
    return np.stack([fwd, left, up])


def local_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


class PhoneArm:
    """Feeds phone packets into an ArmController, handling clutch edges and dropouts."""

    def __init__(self):
        self.arm = ArmController(frame="desk")
        self.arm.M = np.array([[0, 0, -1.0], [-1, 0, 0], [0, 1, 0]])  # until the first engage
        self.engaged_prev = False
        self.last_rx = 0.0
        self.msg: dict | None = None

    def feed(self, msg: dict, now: float) -> None:
        self.msg, self.last_rx = msg, now
        pose = phone_pose(msg)
        engaged = bool(msg.get("engaged")) and msg.get("tracking") == "normal"
        if engaged and not self.engaged_prev:
            self.arm.M = heading_frame(pose, self.arm.M)
            self.arm.engage(pose)
        elif not engaged and self.engaged_prev:
            self.arm.release()
        self.engaged_prev = engaged
        self.arm.grip = pose["grip_open"]  # the gripper follows the tap even while released
        self.pose = pose

    def tick(self, now: float) -> dict:
        if self.msg is not None and now - self.last_rx > 0.3 and self.engaged_prev:
            self.arm.release()  # deadman: the phone went quiet
            self.engaged_prev = False
        fresh = self.msg is not None and now - self.last_rx <= 0.3
        return self.arm.update(self.pose if fresh else None, now)


def run(port: int = 5006, udp: str | None = None) -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", port))
    rx.setblocking(False)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp else None
    dest = (udp.rsplit(":", 1)[0], int(udp.rsplit(":", 1)[1])) if udp else None
    ips = local_ips()
    print(f"Listening for Arm Pilot on UDP {port}. Type one of these into the app: {', '.join(ips) or '(no network?)'}")
    print("Keys: j/l/i/k turn the view, q quit.")
    pa = PhoneArm()
    view = ArmView()
    rate, n, t_rate = 0, 0, time.time()
    while True:
        now = time.time()
        while True:
            try:
                data = rx.recv(4096)
            except BlockingIOError:
                break
            try:
                pa.feed(json.loads(data), now)
                n += 1
            except (ValueError, KeyError):
                continue
        if now - t_rate >= 1:
            rate, n, t_rate = n, 0, now
        st = pa.tick(now)
        if out:
            out.sendto(json.dumps({"t": now, "source": "phone", "arm": st}).encode(), dest)
        if pa.msg is None:
            head = f"waiting for the phone: enter {ips[0] if ips else 'this Mac IP'} port {port} in Arm Pilot"
        elif now - pa.last_rx > 0.3:
            head = "phone went quiet: arm holding"
        else:
            head = f"phone {rate}/s   tracking {pa.msg.get('tracking')}"
        text = [head] + arm_text(st, "lift your thumb to let go")
        if st["engaged"] is False:
            text[1] = "hold your thumb on the phone to move the arm"
        cv2.imshow("rover arm", view.render(pa.arm.q, pa.arm.g, pa.arm.target, text))
        key = cv2.waitKey(10) & 0xFF
        view.key(key)
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
