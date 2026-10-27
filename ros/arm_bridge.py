"""Bridge the operator-side arm model to the rover's joint_trajectory_controller.

Runs on the base station. `stereo-hand wand` (or `phone`, or `mono`) already
runs clutch, inverse kinematics and per-joint speed limits, and sends the
resulting joint angles as JSON over UDP. This node turns each packet into a
one-point JointTrajectory for the controller the team's arm bringup already
starts. Position targets are the safe choice over a radio: if packets stop, the
arm finishes the last target and stays there, where a velocity command would
keep a joint turning.

Joint zero and direction differ between this model and the rover. `offsets`
and `signs` are per joint, in the order base_yaw, shoulder_pitch, elbow_pitch,
wrist_pitch, wrist_roll; they are guesses until measured on the real arm.

    ros2 run ... or: python3 arm_bridge.py --ros-args -p port:=5005
"""

import json
import socket

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
CLAW = ("gripper_claw", -0.045, 0.02)  # closed, open (m), from the MoveIt limits


class ArmBridge(Node):
    def __init__(self):
        super().__init__("operator_arm_bridge")
        self.declare_parameter("port", 5005)
        self.declare_parameter("topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("offsets", [0.0] * 5)
        self.declare_parameter("signs", [1.0] * 5)
        self.declare_parameter("lead_s", 0.1)
        self.pub = self.create_publisher(JointTrajectory, self.get_parameter("topic").value, 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.get_parameter("port").value))
        self.sock.setblocking(False)
        self.create_timer(0.01, self.poll)
        self.sent = 0

    def poll(self):
        latest = None
        while True:
            try:
                latest = self.sock.recv(65535)
            except BlockingIOError:
                break
        if latest is None:
            return
        st = json.loads(latest).get("arm")
        if not st:
            return
        off, sign = self.get_parameter("offsets").value, self.get_parameter("signs").value
        q = [s * v + o for v, s, o in zip(st["q"], sign, off)]
        name, closed, opened = CLAW
        g = closed + (opened - closed) * float(st.get("gripper_open", 1.0))
        lead = self.get_parameter("lead_s").value
        msg = JointTrajectory(joint_names=JOINTS + [name])
        msg.points = [JointTrajectoryPoint(positions=q + [g],
                                           time_from_start=Duration(sec=int(lead), nanosec=int(lead % 1 * 1e9)))]
        self.pub.publish(msg)
        self.sent += 1
        if self.sent % 100 == 1:
            self.get_logger().info("sent %d targets; last %s" % (self.sent, [round(v, 3) for v in q]))


def main():
    rclpy.init()
    rclpy.spin(ArmBridge())


if __name__ == "__main__":
    main()
