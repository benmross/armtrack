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


def test_live_loop_drives_arm_from_video(tmp_path, monkeypatch):
    """The whole `stereo-hand wand` loop on a rendered clip: track every frame, clutch in, the arm follows."""
    import itertools
    import json
    import socket

    import cv2

    from stereo_hand import wand
    from stereo_hand.wand import _exp

    rng = np.random.default_rng(5)
    cube, K = Cube(), camera()
    R0, t0 = random_pose(rng, 700, tilt_deg=15)
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    for i in range(30):
        t = t0 + np.array([120 * i / 29, 0, 0])
        vw.write(cv2.cvtColor(render(cube, R0, t, K, rng), cv2.COLOR_GRAY2BGR))
    vw.release()
    keys = itertools.count()
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: ord("z") if next(keys) == 2 else -1)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    wand.run(path, None, "none", 1, 70.0, f"127.0.0.1:{rx.getsockname()[1]}")
    msgs = []
    try:
        while True:
            msgs.append(json.loads(rx.recv(65535)))
    except OSError:
        pass
    assert len(msgs) == 30 and all(m["pose"] for m in msgs)
    engaged = [m["arm"]["target_tip_m"] for m in msgs if m["arm"]["engaged"]]
    assert len(engaged) >= 26
    moved = np.linalg.norm(np.array(engaged[-1]) - np.array(engaged[0]))
    assert 0.2 < moved < 0.35  # about 115 mm of wand travel at 2.5x


def test_cube_fit_tolerance_and_diagnostics(monkeypatch):
    """A warped face can decode yet fail the cube fit; a demo tolerance is explicit."""
    import cv2
    rng = np.random.default_rng(18)
    cube, K = Cube(), camera()
    R, t = random_pose(rng, 650)
    solver = CubeSolver(K, np.zeros(5), cube)
    gray = render(cube, R, t, K, rng)
    seen = solver.detect(gray)
    assert len(seen) >= 2
    seen = [(mid, pts.copy()) for mid, pts in seen]
    seen[0][1][0] += [12, -8]
    monkeypatch.setattr(solver, 'detect', lambda _: seen)
    solver.max_reproj_px = 0.1
    assert solver.solve(gray) is None
    assert solver.diagnostic['state'] == 'rejected'
    error = solver.diagnostic['reproj_px']
    assert error > 0.1
    solver.max_reproj_px = error + 0.5
    assert solver.solve(gray) is not None
    assert solver.diagnostic['state'] == 'accepted'
    monkeypatch.setattr(solver, 'detect', lambda _: [])
    assert solver.solve(gray) is None
    assert solver.diagnostic == {'state': 'no markers', 'reproj_px': None}
