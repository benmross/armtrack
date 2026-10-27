import math
import struct

import numpy as np

from stereo_hand.wand import PACKET, Cube, CubeSolver, FACES, WandTracker, parse_packets, _rotation_angle
from stereo_hand.wand_sim import camera, random_pose, render


def packet(g, a, trig, btn, us):
    body = struct.pack("<6hHBI", *g, *a, trig, btn, us)
    return b"\xa5\x5a" + body + bytes([sum(body) & 0xFF])


def test_packet_size_matches_firmware():
    assert PACKET.size == 22


def test_parse_resyncs_after_garbage_and_corruption():
    good = packet((655, 0, -655), (0, 0, 8192), 1023, 0b101, 123456)
    bad = bytearray(good)
    bad[5] ^= 0xFF
    buf = bytearray(b"\x00\xa5junk" + bytes(bad) + good + good[:7])
    out = parse_packets(buf)
    assert len(out) == 1
    p = out[0]
    assert math.isclose(math.degrees(p["gyro"][0]), 10.0, rel_tol=1e-3)
    assert math.isclose(p["accel_g"][2], 1.0)
    assert p["clutch"] and not p["home"] and p["fine"] and p["trigger"] == 1.0
    assert bytes(buf) == good[:7]  # the partial packet waits for more bytes


def test_cube_faces_are_outward_and_right_handed():
    cube = Cube()
    for mid, n in FACES.items():
        c = cube.corners(mid)
        u, v = c[1] - c[0], c[3] - c[0]
        assert np.allclose(np.cross(u, v) / np.linalg.norm(np.cross(u, v)), -np.array(n))
        assert np.allclose(c.mean(axis=0), np.array(n) * cube.side_mm / 2)


def test_rendered_cube_solves_to_its_pose():
    rng = np.random.default_rng(0)
    cube, K = Cube(), camera()
    tr = WandTracker(CubeSolver(K, np.zeros(5), cube))
    for _ in range(5):
        R, t = random_pose(rng, 600)
        got = tr.frame(render(cube, R, t, K, rng), 0.0)
        assert got is not None
        assert np.linalg.norm(got[1] - t) < 15
        assert math.degrees(_rotation_angle(R, got[0])) < 3
    assert tr.pose(0.1) is not None and tr.pose(1.0) is None  # held briefly, then lost
