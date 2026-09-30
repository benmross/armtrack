"""The ChArUco calibration board.

ChArUco rather than a plain checkerboard: every corner has an ID, so the two
cameras agree on which corner is which even when each sees only part of the
board, and there is no 180-degree ordering flip between views.
"""

from __future__ import annotations

import cv2
import numpy as np

SQUARES = (7, 5)  # columns, rows
DICT = cv2.aruco.DICT_5X5_100
MARKER_RATIO = 0.75  # marker side / square side


def make_board(square_mm: float) -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard(SQUARES, square_mm, square_mm * MARKER_RATIO, d)


def render(path: str, square_mm: float = 35.0, dpi: int = 300) -> None:
    """Write a print-ready PNG. At 35 mm squares it fits US Letter landscape."""
    board = make_board(square_mm)
    px_per_mm = dpi / 25.4
    margin_mm = 10
    w = int((SQUARES[0] * square_mm + 2 * margin_mm) * px_per_mm)
    h = int((SQUARES[1] * square_mm + 2 * margin_mm) * px_per_mm)
    img = board.generateImage((w, h), marginSize=int(margin_mm * px_per_mm))
    note = f"stereo-hand ChArUco {SQUARES[0]}x{SQUARES[1]}  square {square_mm:g} mm at 100% scale"
    cv2.putText(img, note, (int(margin_mm * px_per_mm), h - int(3 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.imwrite(path, img)


def render_screen(path: str, width: int = 2556, height: int = 1179) -> None:
    """The board filling a screen of the given pixel size, for showing on a
    phone, tablet or laptop instead of printing. Default: iPhone 15/16 Pro in
    landscape. Its physical square size must be measured on the screen."""
    board = make_board(10.0)  # physical size is irrelevant to the image
    margin = int(min(width, height) * 0.04)
    img = board.generateImage((width, height), marginSize=margin)
    cv2.imwrite(path, img)


class Detector:
    def __init__(self, square_mm: float):
        self.board = make_board(square_mm)
        self.det = cv2.aruco.CharucoDetector(self.board)
        self.all_obj = self.board.getChessboardCorners().astype(np.float32)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns (ids (N,), image points (N,2)) or None if too little was seen."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _, _ = self.det.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None
        return ids.ravel().astype(int), corners.reshape(-1, 2).astype(np.float32)
