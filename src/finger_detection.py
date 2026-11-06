import math
import numpy as np


# ----------------------------------------------------------------------
# Landmark indices
# ----------------------------------------------------------------------
WRIST = 0

# (MCP, PIP, DIP, TIP) for each non-thumb finger
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

# ----------------------------------------------------------------------
# Settings (tune these using the calibration window)
# ----------------------------------------------------------------------
# Camera resolution used when image_size is not passed (width, height).
DEFAULT_IMAGE_SIZE = (1280, 720)

# Finger is extended if dist(wrist, tip) / dist(wrist, pip) > REACH_THRESHOLD
REACH_THRESHOLD = 1.05
# ...and the PIP angle is above this (degrees). Kept lenient on purpose.
PIP_ANGLE_MIN = 140

# Thumb is extended if dist(tip, little MCP) / dist(thumb MCP, little MCP)
# is above THUMB_REACH_THRESHOLD and the IP angle is above THUMB_ANGLE_MIN.
THUMB_REACH_THRESHOLD = 1.15
THUMB_ANGLE_MIN = 140


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def to_points(landmarks, image_size=None):
    """
    Convert MediaPipe normalised landmarks to an (21, 3) array in
    pixel-scale units so that angles and distances are not distorted
    by the image aspect ratio. z is scaled like x, as MediaPipe does.
    """
    width, height = image_size if image_size else DEFAULT_IMAGE_SIZE

    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=float
    )


def angle_at(a, b, c):
    """Angle ABC in degrees, where B is the joint (numpy points)."""
    v1 = a - b
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def calculate_angle(a, b, c):
    """
    Kept for compatibility with older code.
    a, b, c are MediaPipe landmarks (raw normalised coordinates).
    """
    pa = np.array([a.x, a.y, a.z])
    pb = np.array([b.x, b.y, b.z])
    pc = np.array([c.x, c.y, c.z])

    return angle_at(pa, pb, pc)


def dist(points, i, j):
    return float(np.linalg.norm(points[i] - points[j]))


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def get_hand_metrics(landmarks, image_size=None):
    """
    Size-independent measurements used for finger and gesture decisions.

    Returns a dictionary with:
        size             palm length (wrist to middle MCP)
        reach            per-finger reach ratio (and "thumb")
        pinch_ratio      thumb tip to index tip distance / palm length
        index_tip_reach  wrist to index tip distance / palm length
    """
    pts = to_points(landmarks, image_size)

    size = max(dist(pts, 0, 9), 1e-6)

    reach = {}

    for finger, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():
        reach[finger] = dist(pts, 0, tip) / max(dist(pts, 0, pip), 1e-6)

    # Thumb: how far the tip is from the little finger base,
    # compared with the thumb MCP.
    reach["thumb"] = dist(pts, 4, 17) / max(dist(pts, 2, 17), 1e-6)

    return {
        "size": size,
        "reach": reach,
        "pinch_ratio": dist(pts, 4, 8) / size,
        "index_tip_reach": dist(pts, 0, 8) / size,
    }


# ----------------------------------------------------------------------
# Finger detection
# ----------------------------------------------------------------------
def detect_fingers(landmarks, image_size=None):
    """
    Detect whether each finger is extended or folded.

    Returns:
        finger_states: dict of True (extended) / False (folded)
        finger_angles: dict of joint angles in degrees
                       (PIP angle for fingers, IP angle for thumb)
    """
    pts = to_points(landmarks, image_size)
    metrics = get_hand_metrics(landmarks, image_size)

    finger_states = {}
    finger_angles = {}

    for finger, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():

        pip_angle = angle_at(pts[mcp], pts[pip], pts[dip])
        finger_angles[finger] = pip_angle

        finger_states[finger] = bool(
            metrics["reach"][finger] > REACH_THRESHOLD
            and pip_angle > PIP_ANGLE_MIN
        )

    # Thumb: angle at the IP joint (2-3-4) plus tip distance check
    thumb_angle = angle_at(pts[2], pts[3], pts[4])
    finger_angles["thumb"] = thumb_angle

    finger_states["thumb"] = bool(
        metrics["reach"]["thumb"] > THUMB_REACH_THRESHOLD
        and thumb_angle > THUMB_ANGLE_MIN
    )

    return finger_states, finger_angles