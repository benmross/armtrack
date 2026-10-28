"""The two interactive loops: calibrate, then track."""

from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from armtrack.vision.board import Detector
from armtrack.vision.calib import Calibration, board_pose, desk_frame, solve
from armtrack.vision.cameras import Source
from armtrack.vision.geometry import OneEuro, hand_pose, match_hands, to_world
from armtrack.vision.hands import HandDetector
from armtrack.arm.kinematics import ArmController
from armtrack.arm.viewer import Viewer
from armtrack.vision.viz import View3D, draw_2d


def _side_by_side(f1: np.ndarray, f2: np.ndarray, w: int = 560) -> np.ndarray:
    a = cv2.resize(f1, (w, int(f1.shape[0] * w / f1.shape[1])))
    b = cv2.resize(f2, (w, int(f2.shape[0] * w / f2.shape[1])))
    h = max(a.shape[0], b.shape[0])
    a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    b = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.hstack([a, b])


def _still(prev, cur, tol: float) -> bool:
    """Is the board in the same place as last frame? The cameras are not
    synchronised, so a pair is only trustworthy if the board was not moving."""
    if prev is None or cur is None:
        return False
    common, i, j = np.intersect1d(prev[0], cur[0], return_indices=True)
    if len(common) < 6:
        return False
    return float(np.linalg.norm(prev[1][i] - cur[1][j], axis=1).mean()) < tol


def _moved(last, cur, min_px: float = 60) -> bool:
    if last is None:
        return True
    return float(np.linalg.norm(last[1].mean(0) - cur[1].mean(0))) > min_px or abs(len(last[0]) - len(cur[0])) > 6


def calibrate(cam1: str, cam2: str, square_mm: float, out: str, target: int = 20,
              width: int = 1280, height: int = 720) -> None:
    s1, s2 = Source(cam1, width, height), Source(cam2, width, height)
    det = Detector(square_mm)
    mono1: list = []
    mono2: list = []
    pairs: list = []
    prev1 = prev2 = last1 = last2 = None
    cov1 = np.zeros((s1.size[1], s1.size[0], 3), np.uint8)
    cov2 = np.zeros((s2.size[1], s2.size[0], 3), np.uint8)
    last_t = 0.0
    auto = True
    print("Move the board slowly through both views: near, far, tilted, every corner of the frame.\n"
          "Hold it still for a moment each time. Space pauses auto-capture, c captures now, "
          "Enter solves, q quits without saving.")
    while True:
        f1, _ = s1.read()
        f2, _ = s2.read()
        if f1 is None or f2 is None:
            break
        r1, r2 = det.detect(f1), det.detect(f2)
        # a hand-held board always shakes a little; allow ~0.4% of the frame width
        st1 = _still(prev1, r1, 0.004 * s1.size[0])
        st2 = _still(prev2, r2, 0.004 * s2.size[0])
        prev1, prev2 = r1, r2
        now = time.time()
        k = cv2.waitKey(1) & 0xFF
        force = k == ord("c")
        if (auto or force) and now - last_t > 0.6:
            took = False
            shared = len(np.intersect1d(r1[0], r2[0])) if (r1 and r2) else 0
            if shared >= 6 and (force or (st1 and st2 and (_moved(last1, r1) or _moved(last2, r2)))):
                pairs.append((r1, r2))
                mono1.append(r1)
                mono2.append(r2)
                last1, last2, took = r1, r2, True
            else:
                if r1 and st1 and _moved(last1, r1) and len(mono1) < 45:
                    mono1.append(r1); last1 = r1; took = True
                if r2 and st2 and _moved(last2, r2) and len(mono2) < 45:
                    mono2.append(r2); last2 = r2; took = True
            if took:
                last_t = now
                for r, cov in ((r1, cov1), (r2, cov2)):
                    if r:
                        for p in r[1]:
                            cv2.circle(cov, tuple(map(int, p)), 6, (0, 90, 0), -1)
        v1 = cv2.addWeighted(f1, 1.0, cov1, 1.0, 0)
        v2 = cv2.addWeighted(f2, 1.0, cov2, 1.0, 0)
        for v, r, st in ((v1, r1, st1), (v2, r2, st2)):
            if r:
                for p in r[1]:
                    cv2.circle(v, tuple(map(int, p)), 4, (0, 255, 255), -1)
            # say what is blocking a capture, big enough to read from arm's length
            label, color = (("NO BOARD", (0, 0, 255)) if not r else
                            (f"{len(r[0])} corners, MOVING", (0, 200, 255)) if not st else
                            (f"{len(r[0])} corners, still", (0, 220, 0)))
            scale = v.shape[1] / 1280
            cv2.rectangle(v, (0, v.shape[0] - int(70 * scale)), (v.shape[1], v.shape[0]), (0, 0, 0), -1)
            cv2.putText(v, label, (int(15 * scale), v.shape[0] - int(20 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5 * scale, color, max(2, int(3 * scale)))
        view = _side_by_side(v1, v2)
        both = "both see it" if (r1 and r2 and len(np.intersect1d(r1[0], r2[0])) >= 6) else "need both to see it"
        msg = (f"pairs {len(pairs)}/{target} ({both})   lens views {len(mono1)} / {len(mono2)}   "
               f"auto {'on' if auto else 'PAUSED'}   Enter solves once pairs >= 8")
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(view, msg, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("armtrack calibrate  (left: cam1, right: cam2; green = covered)", view)
        if k == ord(" "):
            auto = not auto
        if k == ord("q") or k == 27:
            print("quit without saving")
            return
        if k in (13, 10) and len(pairs) < 5:
            print(f"only {len(pairs)} pairs; need at least 5 (8+ is better). Keep going.")
        elif k in (13, 10) or len(pairs) >= target * 2:
            break
    cv2.destroyAllWindows()
    cal = solve(det, mono1, mono2, pairs, s1.size, s2.size, square_mm)
    cal.save(out)
    print(f"saved {out}: cameras are {cal.baseline_mm:.0f} mm apart, stereo error {cal.rms:.2f}px "
          f"({'good' if cal.rms < 1.0 else 'usable' if cal.rms < 2.0 else 'poor: redo with the board held still'})")
    s1.close(); s2.close()


def track(cam1: str, cam2: str, calib_path: str, udp: str | None = None, hand: str | None = None,
          width: int = 1280, height: int = 720, num_hands: int = 2) -> None:
    cal = Calibration.load(calib_path)
    s1, s2 = Source(cam1, width, height), Source(cam2, width, height)
    if s1.size != cal.size1 or s2.size != cal.size2:
        raise SystemExit(f"Camera resolution {s1.size}/{s2.size} differs from calibration "
                         f"{cal.size1}/{cal.size2}. Use the same --width/--height, or recalibrate.")
    d1, d2 = HandDetector(num_hands), HandDetector(num_hands)
    pool = ThreadPoolExecutor(2)
    sock = None
    dest = None
    if udp:
        host, port = udp.rsplit(":", 1)
        dest = (host, int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    desk = not np.allclose(cal.Rw, np.eye(3))
    view = View3D(desk=desk)
    filters = [OneEuro(), OneEuro()]
    zero: np.ndarray | None = None
    rec = None
    det_board = Detector(cal.square_mm)
    arm, arm_view = ArmController(frame="desk" if desk else "camera"), Viewer()
    fps, t_prev = 0.0, time.time()
    print("Keys: z engage the arm (and zero), x release it, the arm is in the browser, b set desk frame from the board lying flat, r record, "
          "a/d/w/s rotate view, +/- zoom, q quit.")
    while True:
        f1, t1 = s1.read()
        f2, t2 = s2.read()
        if f1 is None or f2 is None:
            break
        t = (t1 + t2) / 2
        fut1, fut2 = pool.submit(d1.detect, f1, t1), pool.submit(d2.detect, f2, t2)
        h1, h2 = fut1.result(), fut2.result()
        matched = match_hands(cal, h1, h2, want=hand)
        worlds, text = [], []
        payload = None
        pose = None
        for k, (label, Xc, err) in enumerate(matched[:2]):
            Xw = filters[k](to_world(cal, Xc).ravel(), t).reshape(21, 3)
            worlds.append(Xw)
            if k == 0:
                pose = hand_pose(Xw)
                wrist = np.array(pose["wrist_mm"])
                rel = (wrist - zero) if zero is not None else None
                text += [
                    f"hand {label}  reprojection {err:.1f}px",
                    "wrist  x {:7.1f}  y {:7.1f}  z {:7.1f} mm".format(*wrist),
                    ("from zero  x {:7.1f}  y {:7.1f}  z {:7.1f} mm".format(*rel) if rel is not None
                     else "press z to zero"),
                    f"grip open {pose['grip_open']:.2f}   pinch {pose['pinch_mm']:.0f} mm",
                ]
                payload = {"t": t, "hand": label, "frame": "desk" if desk else "camera1",
                           "reproj_px": err, "relative_mm": rel.tolist() if rel is not None else None,
                           **pose, "landmarks_mm": np.round(Xw, 1).tolist()}
        if payload is None:
            text.append("no hand seen by both cameras" if (h1 or h2) else "no hand")
            payload = {"t": t, "hand": None}
            pose = None
        arm_state = arm.update(pose, t)
        payload["arm"] = arm_state
        if sock:
            sock.sendto(json.dumps(payload).encode(), dest)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        text.append(f"{fps:4.1f} fps   camera skew {abs(t1 - t2) * 1000:3.0f} ms   "
                    f"frame {'desk' if desk else 'camera 1'}{'   REC' if rec else ''}")
        v1, v2 = f1.copy(), f2.copy()
        draw_2d(v1, h1)
        draw_2d(v2, h2)
        cv2.imshow("armtrack cameras  (left: cam1, right: cam2)", _side_by_side(v1, v2))
        cv2.imshow("armtrack 3D", view.render(worlds, text))
        arm_view.publish({**arm_state, "status": ("tracking: " + "stereo cameras" + ("" if arm_state["engaged"] else ", press z to engage"))})
        if rec:
            rec[0].write(f1); rec[1].write(f2)
        key = cv2.waitKey(1) & 0xFF
        view.key(key)
        if key in (ord("q"), 27):
            break
        if key == ord("x"):
            arm.release()
        if key == ord("z") and worlds:
            zero = worlds[0][0].copy()
            arm.engage(hand_pose(worlds[0]))
        if key == ord("b"):
            bp = board_pose(det_board, f1, cal.K1, cal.d1)
            if bp is None:
                print("board not visible to cam1; lay it flat where cam1 can see it")
            else:
                cal.Rw, cal.tw = desk_frame(*bp)
                cal.save(calib_path)
                desk, view.desk, view.center, zero = True, True, None, None
                arm.set_frame("desk")
                view.trail.clear()
                for f in filters:
                    f.x = None
                print("desk frame set: x along the board, y across it, z up. Saved to", calib_path)
        if key == ord("r"):
            if rec:
                rec[0].release(); rec[1].release(); rec = None
                print("recording stopped")
            else:
                Path("recordings").mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                four = cv2.VideoWriter_fourcc(*"mp4v")
                rec = (cv2.VideoWriter(f"recordings/{stamp}-cam1.mp4", four, 30, s1.size),
                       cv2.VideoWriter(f"recordings/{stamp}-cam2.mp4", four, 30, s2.size))
                print(f"recording to recordings/{stamp}-cam*.mp4")
    d1.close(); d2.close(); s1.close(); s2.close()
    cv2.destroyAllWindows()


def calibrate_lens(cam: str, square_mm: float, out: str, target: int = 25,
                   width: int = 1280, height: int = 720) -> None:
    """One camera's lens alone, for the wand, which needs no second camera."""
    from armtrack.vision.calib import _mono
    src = Source(cam, width, height)
    det = Detector(square_mm)
    samples: list = []
    prev = last = None
    last_t = 0.0
    print(f"Move the board slowly through the view: near, far, tilted, every corner. "
          f"{target} still views are taken automatically; Enter solves early, q quits.")
    while len(samples) < target:
        frame, _ = src.read()
        if frame is None:
            break
        r = det.detect(frame)
        still = _still(prev, r, 0.004 * src.size[0])
        prev = r
        if r is not None and still and _moved(last, r) and time.time() - last_t > 0.6:
            samples.append(r)
            last, last_t = r, time.time()
        v = frame.copy()
        if r is not None:
            for p in r[1]:
                cv2.circle(v, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        cv2.putText(v, f"{len(samples)}/{target}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.imshow("lens calibration", v)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            src.close()
            return
        if k == 13:
            break
    src.close()
    cv2.destroyAllWindows()
    if len(samples) < 8:
        raise SystemExit(f"only {len(samples)} views; need at least 8")
    rms, K, d = _mono(det.all_obj, samples, src.size)
    Path(out).write_text(json.dumps({"K": K.tolist(), "d": d.ravel().tolist(), "size": list(src.size), "rms_px": rms}))
    print(f"wrote {out}: {len(samples)} views, reprojection {rms:.2f} px (under 0.5 is good)")
