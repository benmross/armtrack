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


def test_slider_sets_a_partial_grip():
    pa = PhoneArm()
    m = _msg([0, 1, 0], engaged=False)
    m["grip_open"] = 0.35
    _run(pa, [m])
    assert abs(pa.arm.g - 0.35) < 0.01
    m["grip_open"] = 7.0  # nonsense from the wire is clamped
    _run(pa, [m])
    assert pa.arm.g <= 1.0


def test_fine_mode_moves_a_fifth_as_far_and_switching_does_not_jump():
    def travel(fine):
        pa = PhoneArm()
        a = _msg([0, 1, 0]); a["fine"] = fine
        b = _msg([0, 1, -0.08]); b["fine"] = fine
        _run(pa, [a], steps=5)
        tip0 = fk(pa.arm.q)["tip"].copy()
        _run(pa, [b])
        return fk(pa.arm.q)["tip"][0] - tip0[0]
    normal, fine = travel(False), travel(True)
    assert abs(normal - 0.20) < 0.01 and abs(fine - 0.04) < 0.005

    pa = PhoneArm()
    _run(pa, [_msg([0, 1, 0])], steps=5)
    _run(pa, [_msg([0, 1, -0.05])])
    target = pa.arm.target.copy()
    m = _msg([0, 1, -0.05]); m["fine"] = True
    _run(pa, [m], steps=30)  # same place, mode switched mid-hold
    assert np.allclose(pa.arm.target, target, atol=1e-6)


def test_home_button_returns_the_arm_and_needs_a_fresh_hold():
    from stereo_hand.arm import HOME_TIP
    pa = PhoneArm()
    a = _msg([0, 1, 0]); a["home"] = 0
    b = _msg([0.05, 1.05, -0.08]); b["home"] = 0
    _run(pa, [a], steps=5)
    _run(pa, [b])
    assert np.linalg.norm(fk(pa.arm.q)["tip"] - HOME_TIP) > 0.1
    c = dict(b); c["home"] = 1                      # pressed while the thumb is still down
    _run(pa, [c], steps=900)
    assert np.linalg.norm(fk(pa.arm.q)["tip"] - HOME_TIP) < 0.01
    assert not pa.arm.engaged                        # the held thumb did not grab it back
    d = dict(c); d["engaged"] = False
    _run(pa, [d], steps=3)
    _run(pa, [c], steps=3)
    assert pa.arm.engaged                            # lift, then hold again: engaged


def test_warning_only_while_driving():
    pa = PhoneArm()
    far = _msg([0, 1, -2.0])                        # 5 m of arm travel: out of reach
    _run(pa, [_msg([0, 1, 0])], steps=5)
    _run(pa, [far], steps=60)
    assert pa.warning(pa.arm.state()) == "reach"
    off = dict(far); off["engaged"] = False
    _run(pa, [off], steps=5)
    assert pa.warning(pa.arm.state()) == ""


def test_gateways_pick_the_real_router_only(monkeypatch):
    import socket
    from stereo_hand.phone import PAIR_PORT, parse_gateways
    monkeypatch.setattr(socket, "if_nametoindex", lambda n: {"en0": 14, "en7": 21}[n])
    hotspot = """
default            192.0.0.1          UGScg                 en0
default            link#32            UCSIg               utun9
default                                 fe80::7442:18ff:fe41:e364%en0           UGcg                  en0
default                                 fe80::%utun0                            UGcIg               utun0
"""
    assert parse_gateways(hotspot) == [("fe80::7442:18ff:fe41:e364", PAIR_PORT, 0, 14)]
    home = "default            192.168.1.1        UGScg                 en0\n"
    assert parse_gateways(home) == [("::ffff:192.168.1.1", PAIR_PORT, 0, 0)]
