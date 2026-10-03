"""Track a hand in 3-D from two ordinary cameras."""

from __future__ import annotations

import argparse
import json
import socket


def main() -> None:
    p = argparse.ArgumentParser(prog="stereo-hand", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cameras", help="show every camera with its index, to find cam1 and cam2")

    b = sub.add_parser("board", help="write the calibration board as a printable PNG")
    b.add_argument("--square-mm", type=float, default=35.0)
    b.add_argument("--out", default="board.png")
    b.add_argument("--screen", metavar="WxH",
                   help="fill a screen of this many pixels instead, e.g. 2556x1179 for an iPhone 16 in landscape")

    for name in ("calibrate", "track"):
        s = sub.add_parser(name)
        s.add_argument("--cam1", default="0", help="camera index, video file or URL (default 0)")
        s.add_argument("--cam2", default="1")
        s.add_argument("--width", type=int, default=1280)
        s.add_argument("--height", type=int, default=720)
        if name == "calibrate":
            s.add_argument("--square-mm", type=float, required=True,
                           help="measured side of one square on the board as printed/shown, in mm")
            s.add_argument("--out", default="calib.json")
            s.add_argument("--pairs", type=int, default=20)
        else:
            s.add_argument("--calib", default="calib.json")
            s.add_argument("--udp", default="127.0.0.1:5005",
                           help="send one JSON packet per frame here; 'off' to disable")
            s.add_argument("--hand", choices=["Left", "Right"], help="prefer this hand if two are seen")
            s.add_argument("--hands", type=int, default=2, help="max hands to look for per camera")

    m = sub.add_parser("mono", help="track with one camera only (depth from apparent hand size)")
    m.add_argument("--cam", default="1", help="camera index, video file or URL (default 1)")
    m.add_argument("--calib", default="calib.json", help="reuse this camera's lens from a stereo calibration if present")
    m.add_argument("--calib-cam", type=int, choices=[1, 2], default=2,
                   help="which camera in --calib this one was (default 2, the iPhone)")
    m.add_argument("--hfov", type=float, default=70.0, help="horizontal field of view in degrees, if no calibration")
    m.add_argument("--palm-mm", type=float, help="your wrist crease to the base of your middle finger, in mm")
    m.add_argument("--udp", default="127.0.0.1:5005")
    m.add_argument("--width", type=int, default=1280)
    m.add_argument("--height", type=int, default=720)

    lst = sub.add_parser("listen", help="print what track is sending, as an arm controller would receive it")
    lst.add_argument("--port", type=int, default=5005)

    a = p.parse_args()
    if a.cmd == "cameras":
        from .cameras import preview, probe
        found = probe()
        if not found:
            raise SystemExit("No cameras opened. On macOS, give your terminal camera access in "
                             "System Settings > Privacy & Security > Camera.")
        for i, f in found:
            print(f"index {i}: {f.shape[1]}x{f.shape[0]}")
        preview([i for i, _ in found])
    elif a.cmd == "board":
        from .board import render, render_screen
        if a.screen:
            w, h = (int(v) for v in a.screen.lower().split("x"))
            render_screen(a.out, w, h)
            print(f"wrote {a.out}. Show it full screen at maximum brightness, measure one square "
                  "on the screen with a ruler, and pass that to --square-mm.")
            return
        render(a.out, a.square_mm)
        print(f"wrote {a.out}. Print at 100% scale (no 'fit to page'), then measure one square "
              "with a ruler and pass that to --square-mm.")
    elif a.cmd == "calibrate":
        from .app import calibrate
        calibrate(a.cam1, a.cam2, a.square_mm, a.out, a.pairs, a.width, a.height)
    elif a.cmd == "track":
        from .app import track
        track(a.cam1, a.cam2, a.calib, None if a.udp == "off" else a.udp, a.hand, a.width, a.height, a.hands)
    elif a.cmd == "mono":
        from .mono import run
        run(a.cam, a.calib, a.calib_cam, a.hfov, a.palm_mm, None if a.udp == "off" else a.udp, a.width, a.height)
    elif a.cmd == "listen":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", a.port))
        print(f"listening on udp {a.port}")
        while True:
            msg = json.loads(sock.recv(65535))
            if msg.get("hand") is None:
                print("no hand", end="\r")
                continue
            w = msg["relative_mm"] or msg["wrist_mm"]
            print("wrist {:7.1f} {:7.1f} {:7.1f} mm  grip {:.2f}  err {:.1f}px   ".format(
                *w, msg["grip_open"], msg["reproj_px"]), end="\r")
