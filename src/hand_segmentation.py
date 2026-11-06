import math
from collections import deque, Counter

import cv2
import numpy as np

from camera import open_camera
from feature_extraction import extract_fingertips


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
FRAME_WIDTH = 640            # frames are resized to this width for speed
MIN_AREA_RATIO = 0.01        # minimum hand area as a fraction of frame area
BG_THRESHOLD = 25            # background subtraction sensitivity
CLIP_FACTOR = 2.6            # keep only pixels within CLIP_FACTOR * palm radius
DEFECT_DEPTH_FACTOR = 0.30   # minimum defect depth relative to palm radius
DEFECT_MAX_ANGLE = 90        # maximum angle (degrees) between two fingers
DEFECT_MAX_FAR_FACTOR = 2.0  # valley point must be this close to palm centre
SMOOTH_ALPHA = 0.4           # smoothing for palm centre (0..1, higher = faster)


def nothing(_):
    pass


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_between(start, far, end):
    """Angle at 'far' formed by start-far-end, in degrees."""
    a = distance(start, far)
    b = distance(end, far)
    c = distance(start, end)
    if a == 0 or b == 0:
        return 180.0
    cos_val = (a * a + b * b - c * c) / (2 * a * b)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def get_palm(hand_mask):
    """Palm centre and radius using the distance transform."""
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc, max_val


def main():

    # Step 1: Open webcam
    cap = open_camera()

    # Step 2: Create HSV control window with trackbars
    cv2.namedWindow("HSV Controls")
    cv2.createTrackbar("H Min", "HSV Controls", 0, 179, nothing)
    cv2.createTrackbar("H Max", "HSV Controls", 18, 179, nothing)
    cv2.createTrackbar("S Min", "HSV Controls", 55, 255, nothing)
    cv2.createTrackbar("S Max", "HSV Controls", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Controls", 40, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Controls", 255, 255, nothing)

    # Step 3: Morphological kernels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Face detector loading directly from the project root
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    use_face_exclusion = not face_cascade.empty()
    faces = []

    use_ycrcb = True
    background = None
    finger_history = deque(maxlen=7)
    smooth_center = None
    frame_count = 0

    # Info image shown in the controls window
    info = np.zeros((130, 420, 3), np.uint8)
    lines = [
        "q: quit      b: capture background",
        "x: clear bg  f: face exclusion on/off",
        "y: YCrCb check on/off",
        "(move hand out of view before pressing b)",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info, text, (8, 25 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    while True:

        # Step 4: Capture webcam frame
        ret, frame = cap.read()
        if not ret:
            print("Could not read webcam frame")
            break

        # Step 5: Mirror and resize the frame
        frame = cv2.flip(frame, 1)
        h0, w0 = frame.shape[:2]
        scale = FRAME_WIDTH / float(w0)
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h0 * scale)))
        frame_h, frame_w = frame.shape[:2]
        frame_count += 1

        # Step 6: Gaussian blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Step 7: Convert BGR to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Step 8: Read HSV trackbar values
        h_min = cv2.getTrackbarPos("H Min", "HSV Controls")
        h_max = cv2.getTrackbarPos("H Max", "HSV Controls")
        s_min = cv2.getTrackbarPos("S Min", "HSV Controls")
        s_max = cv2.getTrackbarPos("S Max", "HSV Controls")
        v_min = cv2.getTrackbarPos("V Min", "HSV Controls")
        v_max = cv2.getTrackbarPos("V Max", "HSV Controls")

        # Step 9: HSV boundaries
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Step 10: Skin mask (HSV, optionally combined with YCrCb)
        mask = cv2.inRange(hsv, lower, upper)

        if use_ycrcb:
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            ycrcb_mask = cv2.inRange(
                ycrcb, np.array([0, 133, 77]), np.array([255, 180, 135])
            )
            mask = cv2.bitwise_and(mask, ycrcb_mask)

        # Background subtraction (removes the skin-coloured wall)
        if background is not None:
            diff = cv2.absdiff(blurred, background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(
                diff_gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY
            )
            fg_mask = cv2.dilate(fg_mask, kernel_open, iterations=2)
            mask = cv2.bitwise_and(mask, fg_mask)

        # Face exclusion (removes the face and neck from the mask)
        if use_face_exclusion and not face_cascade.empty():
            if frame_count % 5 == 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = face_cascade.detectMultiScale(
                    gray, 1.2, 5, minSize=(60, 60)
                )
                faces = list(found) if len(found) > 0 else []
            for (fx, fy, fw, fh) in faces:
                x1 = max(int(fx - 0.25 * fw), 0)
                y1 = max(int(fy - 0.4 * fh), 0)
                x2 = min(int(fx + 1.25 * fw), frame_w)
                y2 = min(int(fy + 2.2 * fh), frame_h)
                mask[y1:y2, x1:x2] = 0

        # Step 11-12: Remove noise and fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.medianBlur(mask, 5)

        # Step 13: Find contours
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 14: Output frame
        output = frame.copy()
        hand_mask = np.zeros(mask.shape, np.uint8)
        finger_count = None
        defect_count = 0

        # Step 15: Select largest contour
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > MIN_AREA_RATIO * frame_w * frame_h:

                # Filled mask of the largest blob
                cv2.drawContours(hand_mask, [largest], -1, 255, -1)

                # Palm centre and radius
                palm_center, palm_radius = get_palm(hand_mask)

                # Clip forearm / far-away pixels around the palm
                clip = np.zeros_like(hand_mask)
                cv2.circle(
                    clip, palm_center,
                    int(palm_radius * CLIP_FACTOR), 255, -1
                )
                hand_mask = cv2.bitwise_and(hand_mask, clip)

                cnts, _ = cv2.findContours(
                    hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if len(cnts) > 0:
                    contour = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(contour)

                    # Step 16: Extract convex hull and fingertip candidates
                    hull, fingertip_candidates = extract_fingertips(
                        contour,
                        palm_center,
                        palm_radius
                    )

                    # Draw hand contour
                    cv2.drawContours(
                        output,
                        [contour],
                        -1,
                        (0, 255, 0),
                        2
                    )

                    # Draw convex hull
                    cv2.drawContours(
                        output,
                        [hull],
                        -1,
                        (255, 255, 0),
                        2
                    )

                    # Draw detected fingertip candidates
                    for i, tip in enumerate(fingertip_candidates):
                        cv2.circle(
                            output,
                            tip,
                            9,
                            (0, 255, 255),
                            -1
                        )

                        cv2.putText(
                            output,
                            f"Tip {i + 1}",
                            (tip[0] + 10, tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2
                        )

                    # Display number of fingertip candidates
                    cv2.putText(
                        output,
                        f"Tip candidates: {len(fingertip_candidates)}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    # Smooth the palm centre
                    if smooth_center is None:
                        smooth_center = np.array(palm_center, dtype=float)
                    else:
                        smooth_center = (
                            SMOOTH_ALPHA * np.array(palm_center, dtype=float)
                            + (1 - SMOOTH_ALPHA) * smooth_center
                        )
                    cx, cy = int(smooth_center[0]), int(smooth_center[1])

                    # Palm circle and centre
                    cv2.circle(output, (cx, cy), int(palm_radius),
                               (0, 200, 255), 2)
                    cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

                    # Step 17: Convexity defects
                    try:
                        # Get convex hull indices.
                        # Do not sort these indices manually.
                        hull_indices = cv2.convexHull(
                            contour,
                            returnPoints=False
                        )
                        if hull_indices is not None and len(hull_indices) > 3:
                            defects = cv2.convexityDefects(
                                contour,
                                hull_indices
                            )
                            if defects is not None:
                                for i in range(defects.shape[0]):
                                    val = defects[i]
                                    if isinstance(val, np.ndarray):
                                        if val.ndim == 2:
                                            val = val[0]
                                        s_idx, e_idx, f_idx, depth = val[:4]
                                    else:
                                        continue

                                    start = tuple(contour[s_idx][0])
                                    end = tuple(contour[e_idx][0])
                                    far = tuple(contour[f_idx][0])

                                    # OpenCV stores defect depth multiplied by 256.
                                    depth_px = depth / 256.0

                                    angle = angle_between(start, far, end)
                                    far_dist = distance(far, palm_center)

                                    valid = (
                                        depth_px > DEFECT_DEPTH_FACTOR * palm_radius
                                        and angle < DEFECT_MAX_ANGLE
                                        and far_dist < DEFECT_MAX_FAR_FACTOR * palm_radius
                                    )

                                    if valid:
                                        defect_count += 1
                                        # Green: convex hull endpoints
                                        cv2.circle(output, start, 6, (0, 255, 0), -1)
                                        cv2.circle(output, end, 6, (0, 255, 0), -1)
                                        # Red: concave valley between fingers
                                        cv2.circle(output, far, 7, (0, 0, 255), -1)
                                        cv2.line(
                                            output,
                                            start,
                                            end,
                                            (255, 0, 255),
                                            2
                                        )
                    except cv2.error as error:
                        print("Convexity defect error:", error)

                    # Finger count from defects & candidates
                    if defect_count > 0:
                        raw_fingers = min(defect_count + 1, 5)
                    elif len(fingertip_candidates) >= 1:
                        raw_fingers = 1
                    else:
                        raw_fingers = 0

                    finger_history.append(raw_fingers)
                    finger_count = Counter(finger_history).most_common(1)[0][0]

                    # Step 18: Bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(output, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)

                    # Display information
                    cv2.putText(output, f"Area: {int(area)}",
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(output, f"Center: ({cx}, {cy})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(output, f"Defects: {defect_count}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(output, f"Fingers: {finger_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if finger_count is None:
            # No hand found: reset smoothing so the next detection is clean
            finger_history.clear()
            smooth_center = None
            cv2.putText(output, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Status line
        status = (
            f"BG: {'ON' if background is not None else 'OFF'}  "
            f"Face: {'ON' if use_face_exclusion else 'OFF'}  "
            f"YCrCb: {'ON' if use_ycrcb else 'OFF'}"
        )
        cv2.putText(output, status, (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw detected faces being ignored
        for (fx, fy, fw, fh) in (faces if use_face_exclusion else []):
            cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh),
                          (120, 120, 120), 1)

        # Step 20: Segmented region
        segmented = cv2.bitwise_and(frame, frame, mask=hand_mask)

        # Step 21: Display results
        cv2.imshow("Original Webcam", frame)
        cv2.imshow("Binary Mask", mask)
        cv2.imshow("Segmented Hand", segmented)
        cv2.imshow("Hand Detection", output)

        # macOS needs an image shown in the window for trackbars to appear
        cv2.imshow("HSV Controls", info)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            background = blurred.copy()
            print("Background captured")
        elif key == ord('x'):
            background = None
            print("Background cleared")
        elif key == ord('f'):
            if face_cascade.empty():
                print("Face cascade not available")
            else:
                use_face_exclusion = not use_face_exclusion
                faces = []
        elif key == ord('y'):
            use_ycrcb = not use_ycrcb

    # Step 22: Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()