"""A scripted pick and place, for showing the arm without a tracker.

The script only moves the *target*; the arm reaches it through the same IK
and joint speed limits as live control, so the motion is exactly what the
real controller would produce.
"""

from __future__ import annotations

import math
import time

import numpy as np

from armtrack.arm.kinematics import HOME_PITCH, HOME_TIP, ArmController
from armtrack.arm.viewer import Viewer

# tip (m), tool pitch (deg), roll (deg), gripper (1 open, 0 closed), dwell (s), label
SCRIPT = [
    (HOME_TIP, -25, 0, 1, 0.6, "home"),
    ((0.82, -0.28, -0.30), -80, 0, 1, 0.2, "over the rock"),
    ((0.82, -0.28, -0.45), -85, 0, 1, 0.3, "down to it"),
    ((0.82, -0.28, -0.45), -85, 0, 0, 0.9, "grip"),
    ((0.70, -0.20, 0.05), -40, 0, 0, 0.3, "lift"),
    ((0.25, 0.70, 0.10), -40, 45, 0, 0.2, "swing to the side"),
    ((0.25, 0.70, -0.40), -85, 90, 0, 0.3, "lower"),
    ((0.25, 0.70, -0.40), -85, 90, 1, 0.9, "release"),
    ((0.30, 0.55, 0.25), -20, 0, 1, 0.2, "clear"),
    ((0.95, 0.0, 0.45), 10, 0, 1, 0.8, "full reach"),
    ((0.55, 0.0, 0.75), 60, -60, 1, 0.8, "up high, wrist rolled"),
]


def run(view_port: int = 8765, loop: bool = True) -> None:
    viewer = Viewer(view_port)
    arm = ArmController()
    i, settled_at = 0, None
    print("Running the demo script. Ctrl-C to stop.")
    try:
        while True:
            tip, pitch, roll, grip, dwell, label = SCRIPT[i]
            arm.command(np.array(tip, dtype=float), math.radians(pitch), math.radians(roll), grip)
            now = time.time()
            st = arm.update(None, now)
            done = st["lag_m"] < 0.006 and abs(arm.g - arm.grip) < 0.02 and max(st["speed_frac"]) < 0.05
            if done and settled_at is None:
                settled_at = now
            if settled_at is not None and now - settled_at >= dwell:
                i, settled_at = i + 1, None
                if i == len(SCRIPT):
                    if not loop:
                        break
                    i = 0
            viewer.publish({**st, "engaged": True, "status": f"demo: {label} ({i + 1}/{len(SCRIPT)})"})
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
