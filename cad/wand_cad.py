"""Printable wand: a marker cube and a grip that plugs into it.

    uv run --no-project --with trimesh --with manifold3d python cad/wand_cad.py

Writes cad/wand_cube.stl and cad/wand_grip.stl. Print the cube in white PLA:
the plastic around each paper marker is the white quiet zone the detector
needs. Every face has a shallow recess the cut marker squares drop into, which
is what keeps them square and centred; a marker glued 2 degrees crooked costs
more accuracy than anything in the software.

Dimensions in mm. The grip is sized for the Elegoo kit: a GY-521 board flat
under the cube, the kit's joystick module under the thumb (its Y axis opens
and closes the gripper, its press is home), and a 12 mm tactile button under
the index finger as the clutch.
"""

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

SIDE, WALL = 60.0, 2.4
RECESS, RECESS_DEPTH = 56.4, 0.6     # paper squares are cut at 56 mm
PEG_D, PEG_LEN, CLEAR = 24.0, 14.0, 0.25
GRIP_D, GRIP_LEN = 34.0, 120.0
GY521 = (21.5, 16.5, 5.0)            # board plus header stubs, with clearance
WIRE_D = 12.0
OUT = Path(__file__).parent


def cube() -> trimesh.Trimesh:
    body = box((SIDE, SIDE, SIDE))
    body = body.difference(box((SIDE - 2 * WALL,) * 3))
    # recesses on the five marker faces
    for axis, sign in ((2, 1), (0, 1), (0, -1), (1, 1), (1, -1)):
        ext = [RECESS, RECESS, RECESS]
        ext[axis] = RECESS_DEPTH * 2
        r = box(ext)
        off = np.zeros(3)
        off[axis] = sign * SIDE / 2
        r.apply_translation(off)
        body = body.difference(r)
    # bottom: a socket the grip's peg presses into, with a wire hole through the middle
    bottom = box((SIDE, SIDE, WALL))
    bottom.apply_translation((0, 0, -SIDE / 2 + WALL / 2))
    socket_wall = cylinder(radius=PEG_D / 2 + CLEAR + 2.4, height=PEG_LEN)
    socket_wall.apply_translation((0, 0, -SIDE / 2 + PEG_LEN / 2))
    hole = cylinder(radius=PEG_D / 2 + CLEAR, height=PEG_LEN * 3)
    hole.apply_translation((0, 0, -SIDE / 2))
    body = body.union(socket_wall).difference(hole)
    return body


def grip() -> trimesh.Trimesh:
    handle = cylinder(radius=GRIP_D / 2, height=GRIP_LEN, sections=96)
    handle.apply_translation((0, 0, -GRIP_LEN / 2))
    head = box((GRIP_D + 6, GRIP_D + 10, 10))   # flat shelf the IMU sits in, under the cube
    head.apply_translation((0, 0, -5))
    peg = cylinder(radius=PEG_D / 2, height=PEG_LEN, sections=96)
    peg.apply_translation((0, 0, PEG_LEN / 2))
    g = handle.union(head).union(peg)
    # wire channel down the middle, out of the bottom
    ch = cylinder(radius=WIRE_D / 2, height=GRIP_LEN + PEG_LEN + 20)
    ch.apply_translation((0, 0, -GRIP_LEN / 2))
    g = g.difference(ch)
    # IMU pocket in the top of the peg: GY-521 flat, X arrow to the right, Y forward (+y)
    imu = box(GY521)
    imu.apply_translation((0, 0, PEG_LEN - GY521[2] / 2 + 0.01))
    g = g.difference(imu)
    # clutch: 12x12 tactile switch pocket on the front (+y), under the index finger
    clutch = box((12.6, 10, 12.6))
    clutch.apply_translation((0, GRIP_D / 2 - 2, -22))
    g = g.difference(clutch)
    # joystick: the kit module (34x26 mm board, 4 mm holes at 20.5 x 26.5 mm) on a flat cut into the back
    flat = box((GRIP_D + 4, 12, 36))
    flat.apply_translation((0, -GRIP_D / 2 - 2, -30))
    g = g.difference(flat)
    stick = box((18, 20, 22))                    # opening for the stick's gimbal
    stick.apply_translation((0, -GRIP_D / 2 + 6, -30))
    g = g.difference(stick)
    for dx in (-10.25, 10.25):
        for dz in (-13.25, 13.25):
            h = cylinder(radius=1.4, height=14)  # M3 self-tapping pilot holes
            h.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
            h.apply_translation((dx, -GRIP_D / 2 + 4, -30 + dz))
            g = g.difference(h)
    return g


if __name__ == "__main__":
    for name, mesh in (("wand_cube", cube()), ("wand_grip", grip())):
        path = OUT / f"{name}.stl"
        mesh.export(path)
        print(f"{path.name}: {mesh.extents.round(1).tolist()} mm, watertight={mesh.is_watertight}, "
              f"{mesh.volume / 1000:.1f} cm3")
