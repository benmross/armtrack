"""Arm Pilot packets: the phone's motion must move the arm the way the operator means."""

import math

import numpy as np

from stereo_hand.arm import fk
from stereo_hand.phone import PhoneArm


def _msg(pos, yaw_deg=0.0, pitch_deg=0.0, engaged=True, grip=0):
    """A phone held screen-up with its top edge pointing at compass yaw (0 = ARKit -z)
    and tilted up by pitch. Builds ARKit camera axes: x to the bottom edge, y to the
    right side, z out of the screen."""
    yw, pt = math.radians(yaw_deg), math.radians(pitch_deg)
    top = np.array([-math.sin(yw) * math.cos(pt), math.sin(pt), -math.cos(yw) * math.cos(pt)])
    screen = np.array([math.sin(yw) * math.sin(pt), math.cos(pt), math.cos(yw) * math.sin(pt)])
    xc = -top
    zc = screen
    yc = np.cross(zc, xc)
    rot = np.column_stack([xc, yc, zc])
    assert np.isclose(np.linalg.det(rot), 1)
    return {"pos": list(pos), "rot": rot.tolist(), "engaged": engaged, "grip": grip, "tracking": "normal"}


def _run(pa, msgs, t0=0.0, steps=420):
    t = t0
    for m in msgs:
        for _ in range(steps):
            t += 1 / 60
            pa.feed(m, t)
            pa.tick(t)
    return t


def test_push_forward_and_up_whichever_way_you_face():
    for facing in (0.0, 90.0, -135.0):
        pa = PhoneArm()
        tip0 = fk(pa.arm.q)["tip"].copy()
        f = np.array([-math.sin(math.radians(facing)), 0, -math.cos(math.radians(facing))])
        start = np.array([0.1, 1.0, 0.2])
        _run(pa, [_msg(start, facing)], steps=5)                     # engage here
        _run(pa, [_msg(start + 0.08 * f + [0, 0.05, 0], facing)])    # 8 cm forward, 5 cm up
        tip = fk(pa.arm.q)["tip"]
        assert tip[0] - tip0[0] > 0.17, (facing, tip - tip0)
        assert tip[2] - tip0[2] > 0.10, (facing, tip - tip0)
        assert abs(tip[1] - tip0[1]) < 0.02, (facing, tip - tip0)


def test_sideways_is_sideways():
    pa = PhoneArm()
    tip0 = fk(pa.arm.q)["tip"].copy()
    _run(pa, [_msg([0, 1, 0])], steps=5)
    _run(pa, [_msg([0.06, 1, 0])])   # 6 cm to the right when facing ARKit -z
    tip = fk(pa.arm.q)["tip"]
    assert tip[1] - tip0[1] < -0.12  # arm y is left, so right is negative


def test_tilting_up_raises_the_wrist_and_lifting_thumb_holds():
    pa = PhoneArm()
    _run(pa, [_msg([0, 1, 0])], steps=5)
    p0 = pa.arm.pitch
    _run(pa, [_msg([0, 1, 0], pitch_deg=25)])
    assert math.degrees(pa.arm.pitch - p0) > 20
    q = pa.arm.q.copy()
    _run(pa, [_msg([0.3, 1.2, -0.3], engaged=False, grip=1)])
    assert np.allclose(pa.arm.q, q)
    assert pa.arm.g < 0.05  # but the gripper still answers the tap


def test_phone_going_quiet_releases():
    pa = PhoneArm()
    _run(pa, [_msg([0, 1, 0])], steps=5)
    assert pa.arm.engaged
    pa.tick(10.0)
    assert not pa.arm.engaged
