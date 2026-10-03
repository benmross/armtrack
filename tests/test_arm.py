"""The simulated rover arm: kinematics agree, and a hand moving drives the gripper the right way."""

import math

import numpy as np

from stereo_hand.arm import ARM, ArmController, fk, ik


def test_ik_fk_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(200):
        tip = rng.uniform([0.35, -0.5, -0.3], [0.95, 0.5, 0.6])
        pitch = rng.uniform(-1.0, 0.5)
        q, ok = ik(tip, pitch, 0.0)
        if ok:
            assert np.allclose(fk(q)["tip"], tip, atol=1e-6)


def _pose(wrist_mm, grip=0.5, rot=None):
    # a hand facing the camera, fingers up: hand y axis = camera -y, palm out of the screen (camera -z)
    R = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) if rot is None else rot
    return {"wrist_mm": list(wrist_mm), "rotation": R.tolist(), "grip_open": grip}


def test_hand_drives_arm_like_a_clutch():
    arm = ArmController(frame="camera")
    start_tip = fk(arm.q)["tip"].copy()
    t = 0.0
    # before engaging, a moving hand does nothing
    for _ in range(20):
        t += 0.05
        arm.update(_pose([0, 0, 500 - 200 * t]), t)
    assert np.allclose(fk(arm.q)["tip"], start_tip, atol=1e-6)

    arm.engage(_pose([0, 0, 500]))
    # hand 80 mm toward the camera (forward, away from you) and 60 mm up, grip closed
    for _ in range(80):
        t += 0.05
        arm.update(_pose([0, -60, 420], grip=0.0), t)
    tip = fk(arm.q)["tip"]
    assert tip[0] - start_tip[0] > 0.18      # forward by ~2.5 x 80 mm
    assert tip[2] - start_tip[2] > 0.13      # up by ~2.5 x 60 mm
    assert abs(tip[1] - start_tip[1]) < 0.02
    assert arm.g < 0.05

    # release: the arm holds while the hand wanders; re-engage carries on from there
    arm.release()
    held = arm.q.copy()
    for _ in range(10):
        t += 0.05
        arm.update(_pose([300, 300, 900]), t)
    assert np.allclose(arm.q, held)


def test_rate_limit_stops_a_jump():
    arm = ArmController(frame="camera")
    arm.engage(_pose([0, 0, 500]))
    q0 = arm.q.copy()
    arm.update(_pose([0, 0, 500]), 0.0)
    arm.update(_pose([400, -400, 100]), 0.05)  # a tracking glitch: the hand "teleports"
    assert np.abs(np.degrees(arm.q - q0)).max() <= ARM.max_speed_deg * 0.05 + 1e-6
