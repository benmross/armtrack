# stereo-hand

Track a hand in 3-D from two ordinary cameras, as a proof of concept for
controlling a rover arm by moving your own arm instead of a scale-model
controller or a gamepad.

Two cameras look at the same space from different angles (here: a MacBook's
built-in camera and an iPhone through Continuity Camera, one each side of a
desk). MediaPipe finds 21 hand landmarks in each image, the two views are
matched and triangulated using a one-time stereo calibration, and the result is
a hand pose in millimetres: wrist position, orientation, and how open the grip
is. Every frame is also sent as one JSON packet over UDP, which is the seam an
arm controller would plug into.

## How it works

1. **Calibrate once.** Wave a printed ChArUco board through both views. Each
   camera's lens is solved first, then where the cameras sit relative to each
   other (`calib.json`). ChArUco, not a plain checkerboard, so partial views
   still count and the two cameras never disagree on which corner is which.
   The cameras are not hardware-synchronised, so a pair is only captured while
   the board is held still.
2. **Track.** Both frames run through MediaPipe in parallel. Every
   combination of hand-in-camera-1 with hand-in-camera-2 is triangulated and
   the pairing whose 3-D points reproject best into both images wins. That is
   more reliable than trusting left/right labels from a side view. A One Euro
   filter smooths the result without adding lag when you move fast.
3. **Frame.** Output starts in camera-1 coordinates. Lay the board flat on the
   desk and press `b` to switch to a desk frame (x along the board, y across it,
   z up), and press `z` to zero at the current hand position so the output is a
   displacement: the natural input for an arm (a clutch, in teleop terms).

## Setup (macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed
git clone <this repo> && cd stereo-hand
uv sync
```

**iPhone as a camera.** Continuity Camera needs macOS 13+, iOS 16+, the same
Apple ID on both, Wi-Fi and Bluetooth on. Put the phone on a stand in landscape,
rear camera toward you, screen locked. It then appears to the Mac as an ordinary
camera. (Fallback: any iPhone "IP camera" app, and pass its stream URL as
`--cam2`.)

**Camera permission.** Run from Terminal on the Mac itself, not over SSH; the
first run asks to let Terminal use the camera. If it never asks: System
Settings > Privacy & Security > Camera.

## Run

```bash
uv run stereo-hand cameras                  # a labelled tile per camera: note the indices
uv run stereo-hand board                    # board.png: print at 100% scale
uv run stereo-hand calibrate --cam1 0 --cam2 1 --square-mm 35
uv run stereo-hand track --cam1 0 --cam2 1
uv run stereo-hand listen                   # in a second terminal: what an arm would receive
```

`--square-mm` is the side of one square **as measured with a ruler** on your
print; printers scale.

**No printer?** Show the board on any screen that is not one of the two
cameras. `stereo-hand board --screen 2556x1179 --out board-phone.png` fills a
screen of that many pixels (that one is an iPhone 15, 15 Pro or 16 in landscape; use the
device's own resolution). Open it full screen at full brightness, measure a
square on the glass, and pass that as `--square-mm`. A laptop or tablet is
better than a phone: bigger squares detect from further away. Keep glare off
the screen, and expect to hold a phone closer, 25 to 50 cm from each camera.

**Calibration tips.** Aim for a stereo error under 1 px. Hold the board still
for a beat at each pose; tilt it 30 to 45 degrees in different directions;
cover near and far and every corner of both frames (the green overlay shows
coverage). Pairs only count when both cameras see the board, so spend most of
the time in the overlap.

**Placement.** 40 to 70 cm apart with 45 to 90 degrees between their views works
well. Wider angles give better depth but less overlap. After calibrating,
**nothing may move**, including the laptop lid angle. If either camera moves,
recalibrate.

### Track keys

| key | |
|---|---|
| `z` | zero here: output becomes displacement from this point |
| `b` | set the desk frame from the board lying flat where cam1 sees it (saved to calib.json) |
| `r` | record both raw streams to `recordings/` |
| `a` `d` `w` `s` | spin and tilt the 3-D view; `+` `-` zoom |
| `q` | quit |

### Output packet (UDP, default `127.0.0.1:5005`)

```json
{"t": 1790730000.1, "hand": "Right", "frame": "desk", "reproj_px": 2.1,
 "wrist_mm": [x, y, z], "relative_mm": [dx, dy, dz],
 "palm_center_mm": [...], "rotation": [[...]], "quat_wxyz": [...],
 "pinch_mm": 42.0, "grip_open": 0.35, "palm_length_mm": 95.0,
 "landmarks_mm": [[x, y, z], ...21]}
```

`rotation` columns are the hand's axes: x toward the index side, y from wrist
toward the fingers, z out of the palm. `grip_open` is 0 when pinched and 1 when
wide open, normalised by palm size.

## Tests

```bash
uv run pytest
```

These render the board into two simulated cameras 40 cm apart, calibrate with
the real code, and check that the rig, a triangulated hand, and the desk frame
come back within about 1%. They also run MediaPipe on a sample image.

## Limits, and where this goes next

- **Sync.** The cameras free-run, so a fast hand is seen at slightly different
  instants. The skew is shown in the view; it costs accuracy on fast motion.
- **Hand only.** An arm controller probably wants the whole arm (shoulder,
  elbow, wrist) to map joint-for-joint. MediaPipe Pose gives those landmarks
  and triangulates the same way; alternatively, drive the rover arm's inverse
  kinematics from the hand pose alone, which is closer to how it is used.
- **Latency and dropouts.** Before this goes near real hardware it needs a
  deadman (no hand for N frames means hold position), rate and workspace limits,
  and a scale factor between hand motion and arm motion.
