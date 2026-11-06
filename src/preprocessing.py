
import cv2
from camera import open_camera


def main():

    # Open webcam using our camera module
    cap = open_camera()

    while True:

        # Capture a frame
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert BGR image into HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Separate the three HSV channels
        hue, saturation, value = cv2.split(hsv)

        # Display the original image
        cv2.imshow("Original BGR Image", frame)

        # Display individual channels
        cv2.imshow("Hue Channel", hue)
        cv2.imshow("Saturation Channel", saturation)
        cv2.imshow("Value Channel", value)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
