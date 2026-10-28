"""Drawing: landmarks over each camera image, and a rotatable 3-D view of the hand."""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from armtrack.vision.geometry import HAND_EDGES

COLORS = [(80, 200, 255), (255, 160, 80)]


def draw_2d(frame: np.ndarray, hands: list, color=(0, 255, 0)) -> None:
    for _, pts in hands:
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color, 2)
        for p in pts:
            cv2.circle(frame, tuple(map(int, p)), 3, (0, 0, 255), -1)


class View3D:
    """Orthographic view of world points. Up is +z in the desk frame (or -y in
    camera frame, since image y points down). a/d spin, w/s tilt, +/- zoom."""

    def __init__(self, size: int = 640, desk: bool = False):
        self.size = size
        self.desk = desk
        self.yaw, self.pitch, self.scale = 30.0, 20.0, 1.6
        self.center: np.ndarray | None = None
        self.trail: deque = deque(maxlen=90)

    def key(self, k: int) -> None:
        if k == ord("a"): self.yaw -= 8
        elif k == ord("d"): self.yaw += 8
        elif k == ord("w"): self.pitch = min(89, self.pitch + 6)
        elif k == ord("s"): self.pitch = max(-89, self.pitch - 6)
        elif k in (ord("+"), ord("=")): self.scale *= 1.15
        elif k in (ord("-"), ord("_")): self.scale /= 1.15

    def _to_zup(self, X: np.ndarray) -> np.ndarray:
        if self.desk:
            return X
        # camera frame: x right, y down, z forward -> x right, y forward, z up
        return np.stack([X[..., 0], X[..., 2], -X[..., 1]], axis=-1)

    def _project(self, X: np.ndarray) -> np.ndarray:
        Y = self._to_zup(X) - self.center
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        x = cy * Y[..., 0] - sy * Y[..., 1]
        depth = sy * Y[..., 0] + cy * Y[..., 1]
        up = cp * Y[..., 2] - sp * depth
        return np.stack([self.size / 2 + self.scale * x, self.size / 2 - self.scale * up], axis=-1)

    def render(self, hands_xyz: list[np.ndarray], text: list[str]) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 24, np.uint8)
        if hands_xyz:
            c = self._to_zup(hands_xyz[0].mean(0))
            self.center = c if self.center is None else 0.9 * self.center + 0.1 * c
            self.trail.append(hands_xyz[0][0].copy())
        if self.center is not None:
            floor = self.center[2] - 120
            g = np.arange(-300, 301, 50)
            for v in g:
                for a, b in (([v, -300], [v, 300]), ([-300, v], [300, v])):
                    p = np.array([[self.center[0] + a[0], self.center[1] + a[1], floor],
                                  [self.center[0] + b[0], self.center[1] + b[1], floor]])
                    q = self._project_zup(p)
                    cv2.line(img, tuple(map(int, q[0])), tuple(map(int, q[1])), (55, 55, 55), 1)
            if len(self.trail) > 1:
                q = self._project(np.array(self.trail))
                cv2.polylines(img, [q.astype(np.int32)], False, (120, 120, 120), 1)
            for k, X in enumerate(hands_xyz):
                q = self._project(X)
                for a, b in HAND_EDGES:
                    cv2.line(img, tuple(map(int, q[a])), tuple(map(int, q[b])), COLORS[k % 2], 2)
                for p in q:
                    cv2.circle(img, tuple(map(int, p)), 3, (255, 255, 255), -1)
        for i, line in enumerate(text):
            cv2.putText(img, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        return img

    def _project_zup(self, Y: np.ndarray) -> np.ndarray:
        # grid points are already z-up; undo the conversion _project applies
        if self.desk:
            return self._project(Y)
        return self._project(np.stack([Y[:, 0], -Y[:, 2], Y[:, 1]], axis=-1))
