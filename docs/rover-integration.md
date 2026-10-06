# Driving the real arm with Arm Pilot

Notes from reading `umdloop/software-urc4` (main, last commit 16 September
2026) on 6 October 2026. That repository describes itself as the 2025-26 rover,
so details will move, but the structure is what matters here.

## What the rover runs today

- ROS 2 with `ros2_control`. The controller manager runs at 100 Hz on the
  Jetson. The base station runs only the joystick node. The two talk over
  CycloneDDS with fixed peers on 192.168.88.x (`src/config/cyclonedds`).
- The arm in the URDF (`src/description/urdf/arm`): `base_yaw`,
  `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll`, then
  `gripper_claw` (a slider, -0.045 to 0.02 m in the MoveIt limits) and
  `actuator` (a second slider, 0 to 0.05 m). Upper arm and forearm are both
  0.582 m. Every joint's speed limit is 0.5 rad/s, about 29 degrees a second.
- The motors (`src/description/ros2_control/arm`): RMD on shoulder and elbow
  (gear ratio 100), SMC on base yaw (40), three Talon motors on the wrist
  (`wrist_motor_a`, `_b`, `_c`, ratio 102), a servo for `actuator`. Each takes
  position or velocity commands. Base, shoulder and elbow have absolute rotary
  encoders.
- Control (`src/subsystems/arm/arm_controllers`): the joystick publishes
  `sensor_msgs/Joy`; custom controllers turn sticks into joint velocities. The
  active one is joint by joint. A "cylindrical" controller exists that moves
  the tip in a plane using a 3x3 Jacobian. The wrist is driven one motor per
  stick.
- Also configured but not part of manual driving: a
  `joint_trajectory_controller` on position commands, an
  `arm_velocity_controller` (`JointGroupVelocityController`), and a MoveIt
  setup with the KDL solver. `use_mock_hardware:=true` runs the whole arm stack
  with no rover.

Arm Pilot's model already has the same five joints in the same order, so the
idea fits. The link lengths were guessed at 0.52 and 0.56 m and the joint
speeds at 30 to 140 degrees a second; the real values are 0.582 m and 29.

## What has to be built

1. A bridge node on the base station. A small ROS 2 Python node that
   contains what `stereo-hand phone` does now (receive the phone, clutch, IK)
   and publishes joint targets to the rover. Send joint *positions*, to the
   position controller that already exists. A position target is safe when the
   radio drops: the arm finishes the last target and stops. The velocity
   controller keeps its last command when messages stop, so a lost packet
   would leave a joint moving.

2. The real geometry. Replace the estimated planar model with the URDF:
   compute forward kinematics from it so the 3 to 11 cm offsets between joints
   are right, and take limits and speeds from it. The solver itself does not
   change.

3. The wrist mapping. This is the largest unknown. The URDF speaks of
   wrist pitch and roll; the hardware has three wrist motors, and nothing in
   the code converts between them (the controllers drive motors a, b and c
   directly from sticks). The arm's designers have to supply that mapping.

4. Zero positions and directions. For each joint: which way is positive,
   and where is zero, compared with Arm Pilot's convention. Base, shoulder and
   elbow have absolute encoders. The wrist motors do not appear to, so the
   wrist needs a homing step before position commands mean anything.

5. Feedback from the real arm. Subscribe to `/joint_states` so the 3-D
   view shows the real arm next to the commanded one, and so the clutch
   engages from where the arm actually is.

6. Safety on the rover side. The half-second deadman lives on the base
   station now. The rover needs its own: stop if no target has arrived for
   some fraction of a second. Self-collision with the rover body can come from
   MoveIt's planning scene, which the team has configured.

7. Gripper and actuator. Map the phone's 0 to 1 slider onto
   `gripper_claw`'s travel. `actuator` needs a control too, once its purpose is
   confirmed.

The alternative to items 1 and 2 is MoveIt Servo: the phone's motion becomes a
velocity command for the gripper and Servo does the IK, limits and collision
checks with the team's existing MoveIt configuration. It is the standard ROS
route and less code to own. It is not set up in the repository today, and it
still needs items 3 to 7.

## Order of work

1. Bridge node against `use_mock_hardware:=true`, base, shoulder and elbow
   only, watching RViz. No rover needed.
2. Add `/joint_states` feedback and the rover-side deadman.
3. Real arm, same three joints, fine mode, someone on the e-stop.
4. Wrist, once the motor mapping and homing exist.
5. Through the competition radio, where latency and dropouts are real.

Not read for this note: `athena-code` (last year's monorepo), the GUI
repositories, and the firmware.
