"""Measure the wand tracker on rendered images, before any hardware exists.

Each test renders the marker cube at a known pose into a 1280x720 frame from a
70 degree webcam, adds sensor noise and motion blur, runs the real solver on
it, and compares the answer with the pose that drew it. The numbers are
therefore what the geometry and the detector allow; a real camera adds lens
calibration error and lighting, so treat them as a floor.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from armtrack.tracking.wand import FACES, Cube, CubeSolver, OrientationFilter, _exp, _rotation_angle

W, H, HFOV = 1280, 720, 70.0


def camera(w: int = W, h: int = H, hfov: float = HFOV) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def _face_images(cube: Cube, px: int = 360) -> dict[int, np.ndarray]:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = int(round(px * cube.marker_mm / cube.side_mm))
    out = {}
    for mid in FACES:
        tile = np.full((px, px), 235, np.uint8)  # paper is not pure white
        o = (px - m) // 2
        tile[o:o + m, o:o + m] = np.maximum(cv2.aruco.generateImageMarker(d, mid, m, borderBits=1), 25)
        out[mid] = tile
    return out


def render(cube: Cube, R: np.ndarray, t_mm: np.ndarray, K: np.ndarray, rng: np.random.Generator,
           noise: float = 3.0, blur_px: float = 0.0, blur_dir: float = 0.0, faces=None) -> np.ndarray:
    faces = faces or _face_images(cube)
    img = rng.normal(110, 25, (H // 8, W // 8)).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)  # a soft, cluttered background
    for mid, n in FACES.items():
        n = np.asarray(n, float)
        centre = R @ (n * cube.side_mm / 2) + t_mm
        if (R @ n) @ (-centre) <= 0:
            continue  # facing away
        corners = cube.corners(mid, size_mm=cube.side_mm)
        pc = (corners @ R.T + t_mm)
        if np.any(pc[:, 2] < 50):
            continue
        uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
        s = faces[mid].shape[0]
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]), np.float32(uv))
        warped = cv2.warpPerspective(faces[mid], Hm, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), Hm, (W, H))
        shade = 0.55 + 0.45 * abs(float((R @ n)[2]))  # faces turned away from the light are darker
        img = np.where(mask > 128, (warped * shade).astype(np.uint8), img)
    if blur_px >= 1:
        k = int(math.ceil(blur_px)) | 1
        ker = np.zeros((k, k), np.float32)
        c = k // 2
        dx, dy = math.cos(blur_dir), math.sin(blur_dir)
        cv2.line(ker, (int(c - dx * c), int(c - dy * c)), (int(c + dx * c), int(c + dy * c)), 1.0, 1)
        img = cv2.filter2D(img, -1, ker / ker.sum())
    img = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def random_pose(rng: np.random.Generator, dist_mm: float, tilt_deg: float = 45.0):
    """Wand held in front of the camera, pointing roughly away from it, with a random tilt."""
    # camera looks along +z; wand y (forward) along camera +z, wand z (up) along camera -y
    base = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).astype(float)
    tilt = _exp(np.radians(rng.uniform(-tilt_deg, tilt_deg, 3)))
    R = base @ tilt
    t = np.array([rng.uniform(-0.25, 0.25) * dist_mm, rng.uniform(-0.15, 0.15) * dist_mm, dist_mm])
    return R, t


def accuracy(dists=(400, 600, 800, 1000, 1200, 1500), n: int = 60, seed: int = 1, blur_px: float = 0.0):
    rng = np.random.default_rng(seed)
    cube, K = Cube(), camera()
    solver, faces = CubeSolver(K, np.zeros(5), cube), _face_images(cube)
    rows = []
    for dist in dists:
        pe, ae, seen, flips, nf = [], [], 0, 0, []
        for _ in range(n):
            R, t = random_pose(rng, dist)
            img = render(cube, R, t, K, rng, blur_px=blur_px, blur_dir=rng.uniform(0, math.pi), faces=faces)
            got = solver.solve(img)  # no gyro: the worst case for a lone marker
            if got is None:
                continue
            Rh, th, k, _ = got
            a = math.degrees(_rotation_angle(R, Rh))
            if a > 15:
                flips += 1
                continue
            seen += 1
            pe.append(np.linalg.norm(th - t))
            ae.append(a)
            nf.append(k)
        rows.append({"dist_mm": dist, "detected": (seen + flips) / n, "flipped": flips / n,
                     "pos_med_mm": float(np.median(pe)) if pe else None,
                     "pos_p95_mm": float(np.percentile(pe, 95)) if pe else None,
                     "ang_med_deg": float(np.median(ae)) if ae else None,
                     "ang_p95_deg": float(np.percentile(ae, 95)) if ae else None,
                     "faces_mean": float(np.mean(nf)) if nf else None})
    return rows


def jitter(dist_mm: float = 800, n: int = 120, seed: int = 2):
    """Hold the wand perfectly still and measure frame-to-frame scatter: what an operator sees as shake."""
    rng = np.random.default_rng(seed)
    cube, K = Cube(), camera()
    solver, faces = CubeSolver(K, np.zeros(5), cube), _face_images(cube)
    R, t = random_pose(rng, dist_mm, tilt_deg=25)
    P, A = [], []
    for _ in range(n):
        got = solver.solve(render(cube, R, t, K, rng, faces=faces))
        if got:
            P.append(got[1])
            A.append(_rotation_angle(R, got[0]))
    P = np.array(P)
    return {"dist_mm": dist_mm, "frames": len(P), "pos_std_mm": P.std(axis=0).round(2).tolist(),
            "ang_std_deg": float(np.degrees(np.std(A)))}


def blur_sweep(speeds=(0.25, 0.5, 1.0, 2.0), dist_mm: float = 800, exposure_s: float = 1 / 120, n: int = 60):
    """Detection rate when the wand moves sideways at a given hand speed (m/s)."""
    f = camera()[0, 0]
    out = []
    for v in speeds:
        blur = v * 1000 * exposure_s * f / dist_mm
        r = accuracy((dist_mm,), n=n, seed=3, blur_px=blur)[0]
        out.append({"speed_mps": v, "blur_px": round(blur, 1), **r})
    return out


def fusion(seconds: float = 20.0, seed: int = 4, cam_hz: float = 30.0, cam_noise_deg: float = 0.6,
           dropout: float = 0.25, gyro_bias_dps: float = 1.5):
    """Orientation error over a simulated session: camera alone (holding its last fix through
    dropouts), gyro alone, and fused. Camera noise and dropout rate are taken from the image tests."""
    rng = np.random.default_rng(seed)
    dt = 1 / 200
    steps = int(seconds / dt)
    bias = np.radians(rng.normal(0, gyro_bias_dps, 3))
    R = np.eye(3)
    filt, R_gyro, R_cam = OrientationFilter(), np.eye(3), np.eye(3)
    filt.R = np.eye(3)
    occl = (int(8 / dt), int(8.8 / dt))  # one 0.8 s occlusion: the operator's other hand passes in front
    err = {"camera": [], "gyro": [], "fused": []}
    phase = rng.uniform(0, 6, 3)
    for i in range(steps):
        tt = i * dt
        w = np.radians(np.array([90 * math.sin(1.3 * tt + phase[0]), 70 * math.sin(0.9 * tt + phase[1]),
                                 120 * math.sin(1.7 * tt + phase[2])]))  # brisk wrist motion
        R = R @ _exp(w * dt)
        meas = w + bias + np.radians(rng.normal(0, 0.3, 3))  # MPU-6050 noise at 200 Hz
        filt.gyro(meas, dt)
        R_gyro = R_gyro @ _exp(meas * dt)
        if i % int(200 / cam_hz) == 0 and not (occl[0] <= i < occl[1]) and rng.random() > dropout:
            Rm = R @ _exp(np.radians(rng.normal(0, cam_noise_deg, 3)))
            filt.vision(Rm)
            R_cam = Rm
        for k, Rx in (("camera", R_cam), ("gyro", R_gyro), ("fused", filt.R)):
            err[k].append(math.degrees(_rotation_angle(R, Rx)))
    e = {k: np.array(v) for k, v in err.items()}
    win = slice(occl[0], occl[1])
    return {k: {"rms_deg": float(np.sqrt(np.mean(v ** 2))), "during_occlusion_max_deg": float(v[win].max()),
                "final_deg": float(v[-1])} for k, v in e.items()}


def main() -> None:
    import json
    print("accuracy, still wand, no gyro:")
    for r in accuracy():
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    print("jitter, held still at 0.8 m:", json.dumps(jitter()))
    print("motion blur at 0.8 m, 1/120 s exposure:")
    for r in blur_sweep():
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    print("orientation through dropouts:", json.dumps(fusion(), indent=1))


if __name__ == "__main__":
    main()
