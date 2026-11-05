import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Flip image horizontally
    frame = cv2.flip(frame, 1)

    # Display frame
    cv2.imshow("Gesture Recognition - Camera", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()