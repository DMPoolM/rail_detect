import cv2
import numpy as np


LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)


def draw_rails(
    frame: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    thickness: int = 3,
    dot_radius: int = 4,
) -> np.ndarray:
    """Draw left/right rail point series onto a frame (in-place + return).

    left_pts, right_pts: (H, 2) float arrays of (x, y) in frame coordinates.
    left_valid, right_valid: (H,) boolean arrays marking which anchor rows have a rail.
    """
    for pts, valid, color in (
        (left_pts, left_valid, LEFT_COLOR),
        (right_pts, right_valid, RIGHT_COLOR),
    ):
        if valid.sum() >= 2:
            seq = pts[valid].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [seq], isClosed=False, color=color, thickness=thickness)
        for i in range(len(pts)):
            if valid[i]:
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(frame, (x, y), dot_radius, color, -1)
    return frame
