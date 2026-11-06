from collections import deque, Counter


class GestureStabilizer:
    """
    Stabilizes gesture predictions using majority voting
    over a sliding window of recent frames.
    """

    def __init__(self, window_size=7, min_votes=4):
        self.window_size = window_size
        self.min_votes = min_votes

        # Stores the most recent gesture predictions.
        self.gesture_history = deque(maxlen=window_size)

        # Last accepted stable gesture.
        self.stable_gesture = "UNKNOWN"

    def update(self, current_gesture):
        """
        Add the latest prediction and return a stable gesture.
        """

        self.gesture_history.append(current_gesture)

        # Count how many times each gesture appears.
        counts = Counter(self.gesture_history)

        # Find the most frequently predicted gesture.
        most_common_gesture, vote_count = counts.most_common(1)[0]

        # Accept a new gesture only if enough frames agree.
        if vote_count >= self.min_votes:
            self.stable_gesture = most_common_gesture

        return self.stable_gesture

    def reset(self):
        """
        Clear history when no hand is detected.
        """

        self.gesture_history.clear()
        self.stable_gesture = "UNKNOWN"