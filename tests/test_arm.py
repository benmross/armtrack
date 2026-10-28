"""The simulated rover arm: kinematics, robustness, and how it moves."""

import math

import numpy as np

from armtrack.arm.kinematics import ARM, ArmController, JointMotion, clamp_target, fk, ik, solve_ik

LIM = np.radians(np.array(ARM.limits_deg))


def test_ik_reaches_and_respects_limits():
    rng = np.random.default_rng(3)
    hit = 0
    for _ in range(300):
        tip = rng.uniform([0.3, -0.7, -0.4], [1.0, 0.7, 0.8])
        pitch = rng.uniform(-1.4, 0.6)
        q, ok = ik(tip, pitch, rng.uniform(-3, 3))
        assert np.all(q >= LIM[:, 0] - 1e-9) and np.all(q <= LIM[:, 1] + 1e-9)
        if ok:
            hit += 1
            assert np.linalg.norm(fk(q)["tip"] - tip) < 0.005
    assert hit > 200, hit  # most of that box is reachable


def test_ik_path_is_continuous():
    """A straight 1 m sweep in 1 cm steps, warm-started: no joint may jump."""
    q, _ = ik(np.array([0.8, -0.5, 0.0]), -0.5, 0.0)
    for k in range(1, 101):
        tip = np.array([0.8, -0.5 + k * 0.01, 0.0 + 0.3 * math.sin(k / 30)])
        q2, ok = ik(tip, -0.5, 0.0, q0=q)
        assert ok
        assert np.degrees(np.abs(q2 - q)).max() < 6.0, (k, np.degrees(q2 - q))
        q = q2


def test_unreachable_target_settles_at_nearest_pose():
    far = np.array([3.0, 0.0, 0.2])
    q, ok = ik(far, 0.0, 0.0)
    assert not ok
    tip = fk(q)["tip"]
    reach = ARM.upper + ARM.fore + ARM.wrist + ARM.finger
    assert np.linalg.norm(tip - [0, 0, ARM.shoulder_h]) > 0.95 * reach  # stretched toward it
    assert tip[0] > 1.0


def test_targets_stay_out_of_the_ground_and_chassis():
    assert clamp_target([0.8, 0, -2.0])[2] >= ARM.ground_z + ARM.clearance - 1e-9
    p = clamp_target([-0.2, 0.0, -0.1])  # inside the chassis box
    x0, x1, y0, y1, z0, z1 = ARM.chassis
    inside = x0 < p[0] < x1 and y0 < p[1] < y1 and z0 < p[2] < z1
    assert not inside
    assert np.linalg.norm(clamp_target([5, 0, 0]) - [0, 0, ARM.shoulder_h]) <= ARM.upper + ARM.fore + ARM.wrist + ARM.finger


def test_joint_motion_obeys_speed_and_acceleration():
    vmax, amax = np.radians(np.array(ARM.max_speed_deg)), np.radians(np.array(ARM.max_accel_deg))
    m = JointMotion(vmax.copy(), amax.copy())
    q = np.zeros(5)
    goal = np.radians([120, 60, -80, 90, 170])
    dt = 1 / 60
    v_prev = np.zeros(5)
    for _ in range(60 * 12):
        q2 = m.step(q.copy(), goal, dt)
        v = (q2 - q) / dt
        assert np.all(np.abs(v) <= vmax + 1e-6)
        # the last step onto the goal may brake up to ~10% harder than nominal; nowhere else
        assert np.all(np.abs(v - v_prev) <= 1.15 * amax * dt + 1e-6)
        q, v_prev = q2, v
    assert np.allclose(q, goal, atol=1e-6)  # arrives, and does not overshoot past it


def _pose(wrist_mm, grip=1.0):
    R = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return {"wrist_mm": list(wrist_mm), "rotation": R.tolist(), "grip_open": grip}


def test_hand_drives_arm_like_a_clutch():
    arm = ArmController(frame="camera")
    start_tip = fk(arm.q)["tip"].copy()
    t = 0.0
    for _ in range(20):  # not engaged: nothing moves
        t += 0.05
        arm.update(_pose([0, 0, 500 - 200 * t]), t)
    assert np.allclose(fk(arm.q)["tip"], start_tip, atol=1e-6)

    arm.engage(_pose([0, 0, 500]))
    t0 = t
    for _ in range(600):  # 10 s at 60 Hz: 80 mm toward the camera, 60 mm up, grip closed
        t += 1 / 60
        arm.update(_pose([0, -60, 420], grip=0.0), t)
        if t - t0 < 0.3:  # the arm lags rather than jumping
            assert np.linalg.norm(fk(arm.q)["tip"] - start_tip) < 0.08
    tip = fk(arm.q)["tip"]
    assert tip[0] - start_tip[0] > 0.18
    assert tip[2] - start_tip[2] > 0.13
    assert abs(tip[1] - start_tip[1]) < 0.02
    assert arm.g < 0.02

    arm.release()
    held = arm.q.copy()
    for _ in range(10):
        t += 0.05
        arm.update(_pose([300, 300, 900]), t)
    assert np.allclose(arm.q, held, atol=1e-6)


def test_a_tracking_glitch_cannot_fling_the_arm():
    arm = ArmController(frame="camera")
    arm.engage(_pose([0, 0, 500]))
    arm.update(_pose([0, 0, 500]), 0.0)
    q0 = arm.q.copy()
    arm.update(_pose([400, -400, 100]), 1 / 60)  # the hand "teleports"
    acc = np.radians(np.array(ARM.max_accel_deg))
    assert np.all(np.abs(arm.q - q0) <= acc * (1 / 60) ** 2 + 1e-9)
