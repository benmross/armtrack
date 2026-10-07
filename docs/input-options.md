# Choosing the operator's input device

Written 6 October 2026, after the first pitch to the team. Three objections
came up: a phone adds its cost to the rover budget, wireless links may not be
allowed in the control station, and a wired camera watching the hand would
avoid both. Prices are from retailer listings found that day.

## What the rules say

From the URC 2027 rules:

- Budget (3.c.iii): the $26,000 cap covers "all C2 equipment such as
  computers, monitors, controllers". A phone used as a controller counts.
  Used equipment counts at its as-bought price (3.c.vi), and donated
  equipment at the second-hand price of an equivalent (3.c.ix). A refurbished
  iPhone SE (2nd generation), which runs ARKit and the current app, is about
  $100.
- Wireless (2.c): the text does not forbid a wireless link inside the control
  station: the 2.4 GHz band is unrestricted (2.c.vi), and the 900 MHz
  sub-band rules apply to every link including any LAN (2.c.v). The URC2027 Q&A page,
  read 6 October 2026, agrees: Rover Q5 forbids *satellite internet* as a data
  link to the robot, Rover Q11 says there is no internet connection in the
  field, Rover Q6 refers to "the popular 2.4GHz WiFi" band as something teams
  use, and Rover Q7 says 5 GHz and above "is not specifically regulated by
  URC". Nothing there mentions phones, hotspots or cellular. So "no internet"
  is a fact about the site plus a ban on satellite links, and a local Wi-Fi
  link with no internet behind it is allowed. A phone *hotspot* is the
  doubtful case: it is a cellular device, and it may not start at all without
  cell service. Wired is still the better engineering choice. In testing on campus
  Wi-Fi the phone link dropped for 6.5 seconds.

## Options

| Option | Cost | Wired | What it measures | Main risk |
|---|---|---|---|---|
| Marked hand tool and one webcam | $60 to $130 | yes | position, rotation, grip width | needs light and a clear view |
| Leap Motion Controller 2 | $259 | yes | bare hand, 27 joints, true depth | small working volume; vendor changed hands |
| Depth camera with MediaPipe | $200 to $334 | yes | bare hand, measured depth | landmark jitter remains; most code |
| SpaceMouse Compact | $129 | yes | six-axis push and twist, not hand position | rate control, a different feel |
| iPhone on a cable | $100 used, plus adapter | yes | position and rotation | a cable on the hand |
| Two webcams (first prototype) | $40 | yes | bare hand, triangulated | calibration breaks when a camera moves |

### Marked hand tool and one webcam

Square fiducial markers (ArUco) of known printed size on something held in the
hand. One camera recovers full position and rotation from them, because the
marker's true size fixes the distance. That is the difference from tracking a
bare hand with one camera, where distance had to be guessed from how big the
hand looked. Two marked jaws give grip width as well.

Hello Robot sells this as its Dexterous Teleop Kit for the Stretch robot, at
$295:
kitchen tongs with ArUco markers, a Logitech C930e webcam on a stand looking
up, and a ring light so the markers stay sharp during fast motion. Position
drives the arm through inverse kinematics, orientation drives the wrist, and
tong width drives the gripper. The software is open source, so the kit's price buys the
assembled parts and nothing that cannot be put together from an ordinary
webcam, a ring light, tongs and printed markers.

For this project it reuses the arm controller and the 3-D view unchanged, and
OpenCV's ArUco detector is already a dependency. The rover team already
detects the same marker family for the autonomy mission.

### Leap Motion Controller 2

An infrared stereo camera made for hand tracking: 27 joints per hand, USB,
with Linux support in the vendor's software. $259 at Adafruit. Ultraleap,
which makes it, was acquired by ROLI in November 2025, so long-term software
support is a question. It tracks hands in a cone above the sensor, which
suits a seated operator.

### Depth camera with MediaPipe

MediaPipe finds the hand in the colour image and the depth image supplies the
distance, which removes the cause of the single-camera jitter. Intel RealSense
D405 is about $272 and D435i about $334; a Luxonis OAK-D Lite was listed at
about 30,900 yen. The hand's outline still jitters, and this is the most new
code of any option.

### SpaceMouse Compact

A wired six-axis puck, $129. Pushing and twisting it commands velocity in all
six directions at once. It is the default input for the DROID robot dataset
and has ROS drivers. It is not hand tracking, and it is the baseline any
hand-tracking input has to beat: cheap, wired, no line of sight, no lighting.

### iPhone on a cable

The current app, with the phone on a USB-C Ethernet adapter (supported on
iPhone 15 and later with iOS 17) or USB tethering. Tracking quality is the
best of these. A used phone is about $100. Not yet tested wired.

## Recommendation

Build the marked hand tool with one webcam. It answers all three objections,
costs the least of the hand-tracking options, has a commercial precedent, and
needs no new arm code. Keep the SpaceMouse in mind as the fallback.

## Sources

- https://urc.marssociety.org/home/requirements-guidelines/qa
- https://hello-stretch3.com/stretch-dex-teleop-kit
- https://www.adafruit.com/product/5758
- https://github.com/hello-robot/stretch_dex_teleop
- https://hello-robot.com/stretch-dex-teleop-kit
- https://roboticscenter.ai/guides/spacemouse-teleoperation
- https://roboticscenter.ai/learn/robot-camera-depth-sensor-comparison
- https://www.switch-science.com/collections/luxonis/products/7651
- https://apis.io/providers/ultraleap/
- https://www.cultofmac.com/how-to/iphone-ethernet-adapter
- https://macworld.com/article/3217237/this-iphone-se-still-believes-in-buttons-and-its-just-104-99-right-now.html
