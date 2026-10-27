"""Drive the bridge with the operator-side arm model through a scripted reach, for the mock test."""
import json
import socket
import sys
import time

import numpy as np

from stereo_hand.arm import ArmController

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
arm, sock = ArmController(frame="desk"), socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
targets = [(np.array([0.9, 0.3, 0.2]), -0.3, 0.5, 0.2), (np.array([0.7, -0.4, -0.1]), -0.8, -0.6, 1.0)]
t0 = time.time()
for tip, pitch, roll, grip in targets:
    arm.command(tip, pitch, roll, grip)
    end = time.time() + 4.0
    while time.time() < end:
        st = arm.update(None, time.time() - t0)
        sock.sendto(json.dumps({"arm": st}).encode(), ("127.0.0.1", port))
        time.sleep(0.02)
print(json.dumps({"sent_q": st["q"], "gripper_open": st["gripper_open"]}))
