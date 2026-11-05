
import cv2


def main():

    # Open the default webcam
    cap = cv2.VideoCapture(0)

    # Check whether the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    while True:

        # Capture one frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Display the captured frame
        cv2.imshow("Gesture Control System", frame)

        # Exit when the user presses q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
