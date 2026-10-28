"""Drive the rover arm from an iPhone running Arm Pilot (ios/ArmPilot).

The phone tracks its own position and rotation with ARKit, which fuses the
camera with the motion sensors, so position does not drift the way integrating
the accelerometer would, and has none of the depth jitter of guessing distance
from a hand's apparent size. It sends one packet per camera frame (60 Hz).

Holding a thumb on the phone engages the arm (a clutch); lifting it lets go and
the arm holds. "Forward" is wherever the phone's top edge points at the moment
you engage, so you can stand anywhere and face any way.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import time

import numpy as np

from armtrack.arm.kinematics import ArmController
from armtrack.arm.viewer import Viewer


def phone_pose(msg: dict) -> dict:
    """ARKit camera pose -> the pose dict the arm controller takes for a hand.

    ARKit's camera axes: x toward the phone's bottom edge (the sensor is
    landscape-native), y toward its right side, z out of the screen. Held
    screen-up like a remote, the top edge (-x) plays the fingers and the back
    of the phone (-z) plays the palm, which faces down."""
    rot = np.array(msg["rot"], dtype=float)  # columns: camera x, y, z in world (y up)
    xc, yc, zc = rot[:, 0], rot[:, 1], rot[:, 2]
    hand_R = np.column_stack([-yc, -xc, -zc])
    return {"wrist_mm": (np.array(msg["pos"]) * 1000.0).tolist(), "rotation": hand_R.tolist(),
            "grip_open": _grip(msg)}


def _grip(msg: dict) -> float:
    """Gripper opening, 0 closed to 1 open. Older builds of the app sent only a closed flag."""
    if "grip_open" in msg:
        return float(min(1.0, max(0.0, msg["grip_open"])))
    return 0.0 if msg.get("grip") else 1.0


def heading_frame(pose: dict, fallback: np.ndarray) -> np.ndarray:
    """World -> arm rotation with arm x along the phone's current pointing direction, flattened."""
    fwd = np.array(pose["rotation"])[:, 1].copy()
    fwd[1] = 0.0  # ARKit y is up
    n = np.linalg.norm(fwd)
    if n < 0.2:  # pointing nearly straight up or down: heading is meaningless
        return fallback
    fwd /= n
    up = np.array([0.0, 1.0, 0.0])
    left = np.cross(up, fwd)
    return np.stack([fwd, left, up])


def local_addresses() -> list[str]:
    """What to type into the app, most reliable first: this machine's Bonjour
    name (works on any local network, IPv4 or IPv6), then its addresses.
    192.0.0.x is left out: it is a placeholder a Mac gives itself on an
    IPv6-only network such as a phone hotspot, and nothing can reach it."""
    out = []
    name = socket.gethostname().split(".")[0]
    if name:
        out.append(f"{name}.local")
    for family, probe in ((socket.AF_INET, "8.8.8.8"), (socket.AF_INET6, "2001:4860:4860::8888")):
        try:
            s = socket.socket(family, socket.SOCK_DGRAM)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith(("127.", "192.0.0.", "::1", "fe80")):
                out.append(ip)
        except OSError:
            pass
    return out


def open_receiver(port: int) -> socket.socket:
    """One UDP socket that hears both IPv4 and IPv6, so a hotspot that only does IPv6 still works."""
    try:
        rx = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        rx.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        rx.bind(("::", port))
    except OSError:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("0.0.0.0", port))
    return rx


PAIR_PORT = 5007  # Arm Pilot listens here for a Mac announcing itself


def gateways() -> list[tuple]:
    """Socket addresses of this machine's default routers, for an IPv6 dual-stack socket.

    On a phone hotspot (Wi-Fi or USB) the default router *is* the phone, so
    announcing ourselves to the router reaches Arm Pilot with nothing typed.
    On ordinary Wi-Fi the router just ignores it."""
    try:
        table = subprocess.run(["netstat", "-rn"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gateways(table)


def parse_gateways(table: str) -> list[tuple]:
    out = []
    for line in table.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "default":
            continue
        gw, iface = parts[1], parts[-1]
        if iface.startswith(("utun", "lo")) or gw.startswith(("link#", "192.0.0.")):
            continue  # VPN tunnels, and the placeholder router of an IPv6-only network
        try:
            if ":" in gw:
                host, _, scope = gw.partition("%")
                out.append((host, PAIR_PORT, 0, socket.if_nametoindex(scope or iface)))
            elif gw.count(".") == 3:
                out.append((f"::ffff:{gw}", PAIR_PORT, 0, 0))
        except OSError:
            continue
    return list(dict.fromkeys(out))


FINE_SCALE = 0.5
QUIET_S = 0.5  # no packet for this long: the arm lets go (deadman) and the view says so


class LinkLog:
    """Records every gap in the phone's packets, with enough to tell whose fault it was.

    Each packet carries the phone's own frame time. If the phone's clock also
    jumped across a gap, the phone did not produce or send frames (the app
    stalled). If the phone's clock ran normally and the late packets then
    arrive in a clump, the network held them. If they never arrive, the
    network dropped them."""

    def __init__(self, path: str = "phone-link.log", gap_s: float = 0.1):
        self.f = open(path, "a", buffering=1)
        self.gap_s = gap_s
        self.last_rx = self.last_phone_t = None
        self.pending = None   # a gap being characterised: [time, gap, phone gap, packets since, deadline]
        self.worst = 0.0
        self.f.write(f"--- session {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    def packet(self, now: float, phone_t: float | None) -> None:
        if self.pending and now <= self.pending[4]:
            self.pending[3] += 1
        if self.last_rx is not None:
            gap = now - self.last_rx
            if gap > self.gap_s:
                self.flush(now)
                pgap = (phone_t - self.last_phone_t) if (phone_t and self.last_phone_t) else float("nan")
                self.pending = [now, gap, pgap, 0, now + 0.05]
                self.worst = max(self.worst, gap)
        self.last_rx, self.last_phone_t = now, phone_t

    def flush(self, now: float) -> None:
        if self.pending and now > self.pending[4]:
            t, gap, pgap, burst, _ = self.pending
            expect = pgap * 60 if pgap == pgap else 0
            if pgap != pgap:
                verdict = "unknown"
            elif pgap > 0.6 * gap and burst <= 4:
                verdict = "phone sent nothing, or the packets were lost: its clock jumped too"
            elif burst > 4:
                verdict = "network held packets, then delivered them in a clump"
            else:
                verdict = "unclear"
            self.f.write(f"{time.strftime('%H:%M:%S', time.localtime(t))} gap {gap * 1000:.0f} ms, "
                         f"phone clock advanced {pgap * 1000:.0f} ms (~{expect:.0f} frames), "
                         f"{burst} packets in the next 50 ms: {verdict}\n")
            self.pending = None


class PhoneArm:
    """Feeds phone packets into an ArmController, handling clutch edges and dropouts."""

    def __init__(self):
        self.arm = ArmController(frame="desk")
        self.arm.M = np.array([[0, 0, -1.0], [-1, 0, 0], [0, 1, 0]])  # until the first engage
        self.engaged_prev = False
        self.last_rx = 0.0
        self.msg: dict | None = None
        self.home_seen: int | None = None
        self.wait_for_lift = False

    def feed(self, msg: dict, now: float) -> None:
        self.msg, self.last_rx = msg, now
        pose = phone_pose(msg)
        engaged = bool(msg.get("engaged")) and msg.get("tracking") == "normal"
        # fine mode: the arm moves less than the phone does, for lining up on a key or a button
        self.arm.set_scale(FINE_SCALE if msg.get("fine") else self.arm.spec.scale, pose)
        # home is a counter, so a lost or repeated packet cannot miss or repeat the press
        home = msg.get("home")
        if home is not None and self.home_seen is not None and home != self.home_seen:
            self.arm.go_home()
            self.engaged_prev = False
            self.wait_for_lift = True  # a thumb already down must lift before it can grab the arm again
        if home is not None:
            self.home_seen = home
        if not msg.get("engaged"):
            self.wait_for_lift = False
        if self.wait_for_lift:
            engaged = False
        if engaged and not self.engaged_prev:
            self.arm.M = heading_frame(pose, self.arm.M)
            self.arm.engage(pose)
        elif not engaged and self.engaged_prev:
            self.arm.release()
        self.engaged_prev = engaged
        self.arm.grip = pose["grip_open"]  # the gripper follows the tap even while released
        self.pose = pose

    def warning(self, st: dict) -> str:
        """What the phone should buzz about: only while the operator is driving."""
        if not st["engaged"]:
            return ""
        if not st["reachable"]:
            return "reach"
        return "limit" if st["at_limit"] else ""

    def tick(self, now: float) -> dict:
        if self.msg is not None and now - self.last_rx > QUIET_S and self.engaged_prev:
            self.arm.release()  # deadman: the phone went quiet
            self.engaged_prev = False
        fresh = self.msg is not None and now - self.last_rx <= QUIET_S
        return self.arm.update(self.pose if fresh else None, now)


def run(port: int = 5006, udp: str | None = None, view_port: int = 8765) -> None:
    rx = open_receiver(port)
    rx.settimeout(1 / 120)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if udp else None
    dest = (udp.rsplit(":", 1)[0], int(udp.rsplit(":", 1)[1])) if udp else None
    ips = local_addresses()
    print(f"Listening for Arm Pilot on UDP {port}. Type one of these into the app: {', '.join(ips) or '(no network?)'}")
    viewer = Viewer(view_port)
    print("Ctrl-C to stop.")
    pa = PhoneArm()
    link = LinkLog()
    rate, n, t_rate, t_pub, t_fb, phone_addr = 0, 0, time.time(), 0.0, 0.0, None
    hello = json.dumps({"armpilot": 1, "name": socket.gethostname().split(".")[0], "port": port}).encode()
    t_hello, t_gw, gws = 0.0, 0.0, []
    try:
        while True:
            try:
                data, phone_addr = rx.recvfrom(4096)
                msg = json.loads(data)
                link.packet(time.time(), msg.get("t"))
                pa.feed(msg, time.time())
                n += 1
            except (TimeoutError, socket.timeout):
                pass
            except (ValueError, KeyError):
                continue
            now = time.time()
            # nobody talking to us: announce ourselves to the router once a second,
            # from the receiving socket, so the phone can simply answer
            if now - pa.last_rx > 1.0 and now - t_hello >= 1.0:
                t_hello = now
                if now - t_gw > 5.0:
                    t_gw, gws = now, gateways()
                for addr in gws:
                    try:
                        rx.sendto(hello, addr)
                    except OSError:
                        pass
            link.flush(now)
            if now - t_rate >= 1:
                rate, n, t_rate = n, 0, now
            if now - t_pub < 1 / 60:
                continue
            t_pub = now
            st = pa.tick(now)
            if phone_addr and now - t_fb >= 0.1:  # tell the phone, 10 times a second, what to buzz about
                t_fb = now
                try:
                    rx.sendto(json.dumps({"warn": pa.warning(st), "limit": st["at_limit"]}).encode(), phone_addr)
                except OSError:
                    pass
            if out:
                out.sendto(json.dumps({"t": now, "source": "phone", "arm": st}).encode(), dest)
            if pa.msg is None:
                where = ips[1] if len(ips) > 1 else (ips[0] if ips else "this Mac's address")
                status = f"waiting for the phone: on its hotspot it pairs by itself; otherwise enter {where} in Arm Pilot"
            elif now - pa.last_rx > QUIET_S:
                status = "phone went quiet: arm holding"
            else:
                hint = "" if st["engaged"] else ", hold your thumb on the phone to move"
                mode = "fine control 0.5x" if pa.msg.get("fine") else "normal 2.5x"
                limit = f", at limit: {', '.join(st['at_limit'])}" if st["at_limit"] and st["engaged"] else ""
                status = f"phone {rate}/s, {mode}, tracking {pa.msg.get('tracking')}{hint}{limit}"
            viewer.publish({**st, "status": status})
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
