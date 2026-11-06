import os
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from camera import open_camera
from finger_detection import detect_fingers
from gesture_recognition import classify_gesture

# --------------------------------------------------
# 1. Project paths & Settings
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "hand_landmarker.task"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "experiments"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "gesture_frame_results.csv"
)

# Defined gestures for evaluation
GESTURES = [
    "OPEN_PALM",
    "FIST",
    "INDEX_ONLY",
    "TWO_FINGERS",
    "PINCH"
]

# Include UNKNOWN in predictions to ensure unclassified frames count as errors
PREDICTIONS = GESTURES + ["UNKNOWN"]


# --------------------------------------------------
# 2. Main evaluation function
# --------------------------------------------------

def main():

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Hand landmark model not found!")
        print("Expected location:", MODEL_PATH)
        return

    print("Loading MediaPipe Hand Landmarker for Evaluation...")

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

    print("\n--- GESTURE EVALUATION INSTRUCTIONS ---")
    print("Press number keys to select the gesture you are intentionally performing:")
    print("  [1] OPEN_PALM")
    print("  [2] FIST")
    print("  [3] INDEX_ONLY")
    print("  [4] TWO_FINGERS")
    print("  [5] PINCH")
    print("  [0] PAUSE RECORDING (Ignore frames)")
    print("  [q] QUIT AND SAVE RESULTS")
    print("----------------------------------------\n")

    current_expected = "PAUSED"
    frame_index = 0

    # Initialize confusion matrix / tracking structure
    confusion = {
        expected: {predicted: 0 for predicted in PREDICTIONS}
        for expected in GESTURES
    }

    results_data = []

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

            predicted_gesture = "UNKNOWN"

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    finger_states, _ = detect_fingers(landmarks)
                    predicted_gesture = classify_gesture(
                        landmarks,
                        finger_states
                    )

            # Record result if actively testing a valid gesture
            if current_expected in GESTURES and predicted_gesture in PREDICTIONS:
                confusion[current_expected][predicted_gesture] += 1
                results_data.append({
                    "timestamp": time.time(),
                    "expected": current_expected,
                    "predicted": predicted_gesture
                })

            # Display status on screen
            cv2.putText(
                frame,
                f"Expected (Target): {current_expected}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                f"Predicted: {predicted_gesture}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gesture Evaluation", frame)

            # Keyboard input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                current_expected = "OPEN_PALM"
                print("Target gesture set to: OPEN_PALM")
            elif key == ord("2"):
                current_expected = "FIST"
                print("Target gesture set to: FIST")
            elif key == ord("3"):
                current_expected = "INDEX_ONLY"
                print("Target gesture set to: INDEX_ONLY")
            elif key == ord("4"):
                current_expected = "TWO_FINGERS"
                print("Target gesture set to: TWO_FINGERS")
            elif key == ord("5"):
                current_expected = "PINCH"
                print("Target gesture set to: PINCH")
            elif key == ord("0"):
                current_expected = "PAUSED"
                print("Evaluation paused.")

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

        # Save results to CSV file
        if results_data:
            with open(OUTPUT_FILE, mode="w", newline="") as csv_file:
                fieldnames = ["timestamp", "expected", "predicted"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_data:
                    writer.writerow(row)
            print(f"\nResults successfully saved to {OUTPUT_FILE}")

        # Compute and print overall accuracy metrics
        total_frames = 0
        total_correct = 0

        print("\n--- EVALUATION SUMMARY ---")
        for expected in GESTURES:
            row_total = sum(confusion[expected].values())
            row_correct = confusion[expected][expected]
            total_frames += row_total
            total_correct += row_correct

            acc = (row_correct / row_total * 100) if row_total > 0 else 0.0
            print(f"Gesture '{expected}': {row_correct}/{row_total} correct ({acc:.2f}%)")

        overall_acc = (total_correct / total_frames * 100) if total_frames > 0 else 0.0
        print(f"\nOverall Frame-Level Accuracy: {total_correct}/{total_frames} ({overall_acc:.2f}%)")
        print("Program closed successfully.")


if __name__ == "__main__":
    main()