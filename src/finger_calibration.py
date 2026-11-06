import os
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import open_camera
from finger_detection import detect_fingers, get_hand_metrics
from gesture_recognition import classify_gesture, GestureSmoother


# Locate the MediaPipe hand landmark model
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "hand_landmarker.task"
)


def main():

    if not os.path.exists(MODEL_PATH):
        print("Hand landmark model not found:", MODEL_PATH)
        return

    # Initialize MediaPipe
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = open_camera()

    if cap is None:
        landmarker.close()
        return

    smoother = GestureSmoother(size=7)
    start_time = time.monotonic()
    last_timestamp = -1

    print("\nFinger calibration started.")
    print("Show different hand gestures to the camera.")
    print("Press Q to quit.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            image_size = (frame_w, frame_h)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamps must be strictly increasing
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1
            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                finger_states, finger_angles = detect_fingers(
                    landmarks, image_size
                )

                metrics = get_hand_metrics(landmarks, image_size)

                raw_gesture = classify_gesture(
                    landmarks, finger_states, image_size
                )
                gesture = smoother.update(raw_gesture)

                cv2.putText(
                    frame, f"Gesture: {gesture}  (raw: {raw_gesture})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

                # Angle, reach ratio and state for each finger
                y = 80

                for finger in ["thumb", "index", "middle", "ring", "little"]:

                    angle = finger_angles[finger]
                    reach = metrics["reach"][finger]
                    status = "EXTENDED" if finger_states[finger] else "FOLDED"

                    text = (
                        f"{finger.upper()}: {angle:.0f} deg  "
                        f"reach {reach:.2f}  {status}"
                    )

                    cv2.putText(
                        frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )

                    y += 32

                # Values used for PINCH decisions
                cv2.putText(
                    frame,
                    f"Pinch ratio: {metrics['pinch_ratio']:.2f}   "
                    f"Index reach: {metrics['index_tip_reach']:.2f}",
                    (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2
                )

            else:

                smoother.reset()

                cv2.putText(
                    frame, "NO HAND DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow("Finger Angle Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

        print("Calibration program closed.")


if __name__ == "__main__":
    main()