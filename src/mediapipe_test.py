import os
import cv2
import mediapipe as mp
from camera import open_camera
from mouse_control import MouseController
from gesture_recognition import classify_gesture
from finger_detection import detect_fingers
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. Locate the model file
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "hand_landmarker.task"
)


# --------------------------------------------------
# 2. Define connections between hand landmarks
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# 3. Draw landmarks and connections
# --------------------------------------------------

def draw_hand_landmarks(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for index, point in enumerate(points):
        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )
        cv2.putText(
            frame,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1
        )


# --------------------------------------------------
# 4. Main function
# --------------------------------------------------

def main():

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Hand landmark model not found!")
        print("Expected location:", MODEL_PATH)
        return

    print("Loading MediaPipe Hand Landmarker...")

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = open_camera()

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        landmarker.close()
        return

    print("Webcam started successfully.")
    print("Show your hand in front of the camera.")
    print("Press Q to quit.")

    frame_index = 0
    mouse_controller = MouseController()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = frame_index * 33
            frame_index += 1

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:
                for hand_number, landmarks in enumerate(
                    result.hand_landmarks
                ):
                    finger_states, finger_angles = detect_fingers(landmarks)
                    
                    gesture = classify_gesture(
                        landmarks,
                        finger_states
                    )

                    # Execute the mouse action corresponding to the gesture
                    # Execute the mouse action corresponding to the gesture
                    mouse_controller.handle_gesture(gesture, landmarks, finger_states)

                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    y_position = 140

                    for finger, state in finger_states.items():
                        status = "Extended" if state else "Folded"
                        text = f"{finger}: {status}"

                        cv2.putText(
                            frame,
                            text,
                            (20, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2
                        )
                        y_position += 30

                    draw_hand_landmarks(frame, landmarks)

                    cv2.putText(
                        frame,
                        "Hand Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    if result.handedness:
                        handedness = (
                            result.handedness[hand_number][0].category_name
                        )
                        cv2.putText(
                            frame,
                            handedness,
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )
            else:
                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "MediaPipe Hand Landmark Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("Program closed successfully.")


if __name__ == "__main__":
    main()