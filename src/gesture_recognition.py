from collections import Counter, deque

from finger_detection import get_hand_metrics


# ----------------------------------------------------------------------
# Settings (tune these using the calibration window)
# ----------------------------------------------------------------------
# Thumb tip to index tip distance (relative to palm length) for a pinch
PINCH_RATIO = 0.30

# In a pinch the index tip stays away from the palm; in a fist it does not
PINCH_INDEX_REACH = 1.15


def classify_gesture(landmarks, finger_states, image_size=None):
    """
    Classify the hand into one of:
    PINCH, FIST, OPEN_PALM, INDEX_ONLY, TWO_FINGERS, UNKNOWN

    The thumb is ignored for everything except PINCH because it is the
    least reliable finger.
    """
    metrics = get_hand_metrics(landmarks, image_size)

    index = finger_states["index"]
    middle = finger_states["middle"]
    ring = finger_states["ring"]
    little = finger_states["little"]

    # Pinch is checked first: it also has a curled index finger,
    # so it would otherwise be mistaken for a fist.
    if (
        metrics["pinch_ratio"] < PINCH_RATIO
        and metrics["index_tip_reach"] > PINCH_INDEX_REACH
    ):
        return "PINCH"

    extended_count = sum([index, middle, ring, little])

    if extended_count == 0:
        return "FIST"

    if index and middle and ring and little:
        return "OPEN_PALM"

    if index and middle and not ring and not little:
        return "TWO_FINGERS"

    if index and not middle and not ring and not little:
        return "INDEX_ONLY"

    return "UNKNOWN"


class GestureSmoother:
    """Majority vote over the last few frames to remove flicker."""

    def __init__(self, size=7):
        self.history = deque(maxlen=size)

    def update(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]

    def reset(self):
        self.history.clear()