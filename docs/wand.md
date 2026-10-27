# The marker-cube wand

Written 27 October 2026 for the software meeting. This is a proposal for the
arm's hand-movement controller, with the parts of it that can be built
without hardware already built and measured.

## What it is

A 3-D printed handle with a 60 mm white cube on top. Five faces of the cube
each carry a printed 50 mm ArUco marker. One ordinary USB webcam on the
operator's desk watches it. Because each marker's real size is known, a
single camera recovers the handle's full position and rotation; nothing has
to be guessed from how big a hand looks. A gyro (MPU-6050) inside the handle
streams rotation rates at 200 Hz to an Arduino, which also reads a joystick
for the gripper and buttons for clutch, home and fine mode. Everything is
wired: the camera and the Arduino are two USB cables into the base station.

The operator holds the clutch and moves. The arm's gripper follows the
handle's displacement (scaled 2.5x, or 0.6x in fine mode) and its pitch and
roll follow the handle's, through the inverse kinematics and per-joint speed
limits Arm Pilot already has. Letting go of the clutch parks the arm.

## Why this design

The team chose hand movement over a scale-model leader arm, which it
estimated at $500 or more. The objections to the iPhone version were its cost
on the competition budget and the wireless link. This design answers both:
under $50 in parts, wired end to end, and built from parts the team can reuse.

The two sensors cover each other's weak points:

- The camera gives absolute position and rotation, but only about 30 times a
  second, and it fails when the cube blurs or is hidden.
- The gyro gives rotation 200 times a second and never loses sight of
  anything, but it drifts by about a degree a second.

The software fuses them: the gyro carries rotation between and through camera
fixes, and each camera fix corrects the gyro's drift and learns its bias. The
gyro also settles the main weakness of a single flat marker, which fits two
mirror-image rotations nearly equally well when only one face is visible.

Position comes from the camera only. No IMU can hold position: integrating
acceleration twice drifts by centimetres within a second. If the camera loses
the cube for more than a quarter second, the controller lets go of the arm.

## Measured so far (no hardware)

`stereo-hand wand-sim` renders the cube at known poses into 1280x720 frames
from a 70-degree webcam, adds sensor noise and motion blur, and runs the real
tracking code on them. These are therefore the best case the geometry allows.
A real camera adds lens calibration error.

| Distance | Median position error | 95th percentile | Median angle error | Wrong-flip rate |
|---|---|---|---|---|
| 0.4 m | 1.0 mm | 2.4 mm | 0.3° | 0% |
| 0.6 m | 1.7 mm | 5.0 mm | 0.4° | 0% |
| 0.8 m | 4.2 mm | 9.4 mm | 0.5° | 0% |
| 1.0 m | 4.4 mm | 21 mm | 0.8° | 0% |
| 1.2 m | 8.2 mm | 25 mm | 0.8° | 3% |
| 1.5 m | 17 mm | 34 mm | 1.3° | 13% |

Those errors are absolute, and most of them are in depth. The arm is driven
by displacement from the moment the clutch closes, so what the operator feels
is frame-to-frame jitter. Held still at 0.8 m that is 0.1 to 0.2 mm in
position and 0.13° in angle, or about 0.5 mm at the gripper after scaling.
The camera should sit 0.5 to 0.8 m from where the hand works.

Motion blur is the real limit. At 0.8 m with a 1/120 s exposure, the tracker
holds up to 0.5 m/s of hand speed, is degraded at 1 m/s (20% of frames picked
the wrong flip with no gyro), and loses the cube entirely at 2 m/s. Arm
joints are capped at 29°/s, so the operator never needs to move fast, but the
camera should be set to a short manual exposure with a desk lamp or ring
light on the cube.

With the gyro, a simulated 20-second session of brisk wrist motion, including
a 0.8 s occlusion and 25% dropped frames, kept orientation within 0.42° RMS
and 0.76° worst case during the occlusion. The camera alone, holding its
last good fix, was off by up to 107° in the same occlusion; the gyro alone
had drifted to 7.5° by the end.

## What is built

All of it is in this repository, on top of the existing arm model.

- `src/stereo_hand/wand.py`: marker detection, a joint solve over every
  visible corner, gyro fusion, the serial reader for the grip, and
  `stereo-hand wand`, the live loop that drives the 3-D arm view and sends
  the arm's joint angles over UDP.
- `src/stereo_hand/wand_sim.py`: the measurements above.
- `firmware/wand_grip/wand_grip.ino`: reads the MPU-6050, joystick and
  buttons, and sends checksummed 22-byte packets at 200 Hz. It compiles for
  an Uno (14% of flash) and a Nano.
- `cad/wand_cad.py` writes `wand_cube.stl` and `wand_grip.stl`. The cube has
  a shallow recess on each face that the cut-out markers drop into, which
  keeps them square. The grip has a pocket for the GY-521 under the cube, a
  flat for the kit's joystick module under the thumb, and a pocket for a
  12 mm button under the index finger. These dimensions are first guesses
  and need one test print.
- `stereo-hand wand-markers` writes the printable page of markers.
- `ros/arm_bridge.py`: a ROS 2 node that turns those joint angles into
  `JointTrajectory` targets for the `joint_trajectory_controller` the arm
  bringup already configures. `ros/mock_arm.sh` starts software-urc4's own
  arm URDF on mock hardware. Driving it through the bridge put base yaw,
  shoulder, elbow, wrist pitch, wrist roll and the claw exactly on the
  commanded values.

## Parts

| Part | Source | Cost |
|---|---|---|
| Arduino Uno, joystick module, buttons, jumpers | Elegoo starter kit | owned |
| MPU-6050 (GY-521) | not found in the kit's published parts list; check for it, otherwise a multipack | a few dollars each |
| USB webcam, 720p or better, manual exposure (Logitech C270 or similar) | any retailer | about $30 |
| Cube and grip | X1 Carbon, white PLA for the cube | filament |
| Markers | paper printer, glued into the recesses | paper |
| 4-conductor cable, about 1 m, for the grip (I2C to the gyro), plus 4 more for buttons and joystick | team stock or a USB cable cut open | none |

An Arduino Nano clone in the grip would replace the long I2C run with a
single USB cable. The firmware is the same.

For competition, two upgrades are worth their cost: a global-shutter USB
camera, which removes motion blur, and a second camera at a right angle,
which removes the depth error. The total stays far under $500.

## Alternatives considered

- **Bare-hand tracking** (the first stereo-hand prototype, MediaPipe): no
  held object, but landmark jitter is several millimetres and depth from one
  camera depends on hand size.
- iPhone (Arm Pilot): already works and tracks well, but it counts at
  its price on the budget and was built around Wi-Fi.
- SteamVR tracker on a USB cable: sub-millimetre and tolerant of occlusion, but a
  tracker and two base stations cost several hundred dollars, and the
  open-source driver (libsurvive) is less mature than the markers.
- Leap Motion Controller 2: bare hand with true depth, $259, a small
  working volume, and uncertain long-term vendor support.
- IMUs strapped to the upper arm, forearm and hand: no line of sight is
  needed, but strap slip and the MPU-6050's lack of a magnetometer make
  heading drift, and position error grows along the chain.
- Three tethers to the handle with encoders on their reels: absolute,
  cheap and immune to light, but the cables get in the operator's way.

## To try with the hardware

1. Print the cube and markers. Check one black square measures 50.0 mm.
2. Run `stereo-hand wand --cam 0 --hfov <the webcam's field of view>` with
   no grip. The keyboard's z and x act as the clutch. A guessed lens costs
   depth accuracy; `calibrate` currently needs two cameras, so a
   single-camera lens calibration is the next thing to write.
3. Wire the GY-521, joystick and buttons per the firmware's header comment,
   flash it, and add `--serial /dev/ttyACM0`. Hold the wand still for the
   first second while the gyro bias is measured.
4. On a machine with ROS 2 Jazzy, run `ros/mock_arm.sh` and
   `ros/arm_bridge.py`, then drive the arm in RViz.

## Open questions for the team

- Joint zero positions and directions on the real arm, compared with this
  model's (`offsets` and `signs` on the bridge).
- How the wrist's three motors map to wrist pitch and roll. Nothing in
  software-urc4 converts between them yet.
- Whether the base station can host one more USB camera pointed at the
  operator, and where it would sit in the control station.
