
import os
import time
import csv
from datetime import datetime
from collections import Counter, deque

import tkinter as tk
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import open_camera
from virtual_keyboard import VirtualKeyboard
from mouse_control import MouseController
from finger_detection import detect_fingers
from gesture_recognition import classify_gesture
from gesture_stabilizer import GestureStabilizer


# --------------------------------------------------
# 1. Paths and MediaPipe configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

EXPERIMENTS_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# 2. Draw hand landmarks
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

    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


# --------------------------------------------------
# 3. Main dashboard application
# --------------------------------------------------

class GestureKeyboardApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Gesture Control System")
        self.root.geometry("1150x780")
        self.root.minsize(850, 550)

        self.running = True
        self.camera_active = False
        self.mouse_enabled = False

        self.cap = None
        self.landmarker = None

        self.keyboard_window = None
        self.keyboard = None

        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

        # Performance monitoring variables.

        self.total_frames = 0
        self.hand_detected_frames = 0

        self.gesture_counts = Counter()

        self.frame_times = deque(maxlen=30)
        self.processing_times = deque(maxlen=60)

        self.session_saved = False

        # Initialize controllers.

        self.mouse_controller = MouseController()

        self.gesture_stabilizer = GestureStabilizer(
            window_size=7,
            min_votes=4
        )

        self.initialize_mediapipe()

        # Create scrollable dashboard.

        self.create_scrollable_container()

        # Create all dashboard widgets inside it.

        self.create_dashboard()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        print("Gesture Control Dashboard initialized.")

    # --------------------------------------------------
    # 4. Create scrollable dashboard container
    # --------------------------------------------------

    def create_scrollable_container(self):

        # Outer frame fills the entire application window.

        self.outer_frame = tk.Frame(self.root)

        self.outer_frame.pack(
            fill="both",
            expand=True
        )

        # Canvas provides the scrolling area.

        self.canvas = tk.Canvas(
            self.outer_frame,
            highlightthickness=0
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        # Vertical scrollbar.

        self.scrollbar = tk.Scrollbar(
            self.outer_frame,
            orient="vertical",
            command=self.canvas.yview
        )

        self.scrollbar.pack(
            side="right",
            fill="y"
        )

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set
        )

        # This frame will contain all dashboard widgets.

        self.dashboard_frame = tk.Frame(
            self.canvas
        )

        self.dashboard_window = self.canvas.create_window(
            (0, 0),
            window=self.dashboard_frame,
            anchor="nw"
        )

        # Update scrollable area when its contents change.

        self.dashboard_frame.bind(
            "<Configure>",
            self.update_scroll_region
        )

        # Keep dashboard width equal to canvas width.

        self.canvas.bind(
            "<Configure>",
            self.resize_dashboard
        )

        # Mouse-wheel scrolling.

        self.canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel
        )

        # Linux mouse-wheel support.

        self.canvas.bind_all(
            "<Button-4>",
            self.on_mousewheel
        )

        self.canvas.bind_all(
            "<Button-5>",
            self.on_mousewheel
        )

    def update_scroll_region(self, event=None):

        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def resize_dashboard(self, event):

        self.canvas.itemconfigure(
            self.dashboard_window,
            width=event.width
        )

    def on_mousewheel(self, event):

        if event.num == 4:

            self.canvas.yview_scroll(-1, "units")

        elif event.num == 5:

            self.canvas.yview_scroll(1, "units")

        else:

            # macOS and Windows mouse-wheel support.

            if event.delta > 0:

                self.canvas.yview_scroll(-1, "units")

            elif event.delta < 0:

                self.canvas.yview_scroll(1, "units")

    # --------------------------------------------------
    # 5. Initialize MediaPipe
    # --------------------------------------------------

    def initialize_mediapipe(self):

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"MediaPipe model not found: {MODEL_PATH}"
            )

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

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

    # --------------------------------------------------
    # 6. Create dashboard interface
    # --------------------------------------------------

    def create_dashboard(self):

        # IMPORTANT:
        # All widgets are now children of dashboard_frame,
        # not the root window.

        parent = self.dashboard_frame

        # Main heading.

        heading = tk.Label(
            parent,
            text="GESTURE CONTROL SYSTEM",
            font=("Arial", 24, "bold")
        )

        heading.pack(pady=(15, 3))

        subtitle = tk.Label(
            parent,
            text="Control your computer using hand gestures",
            font=("Arial", 12)
        )

        subtitle.pack(pady=(0, 12))

        # Main content container.

        content = tk.Frame(parent)

        content.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        # ----------------------------------------------
        # Left: Camera preview
        # ----------------------------------------------

        camera_frame = tk.LabelFrame(
            content,
            text="Live Camera Preview",
            font=("Arial", 14, "bold"),
            padx=10,
            pady=10
        )

        camera_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=8
        )

        self.camera_label = tk.Label(
            camera_frame,
            text="Camera is currently stopped",
            font=("Arial", 14),
            width=55,
            height=20,
            bg="black",
            fg="white"
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # ----------------------------------------------
        # Right: Status and performance
        # ----------------------------------------------

        status_frame = tk.LabelFrame(
            content,
            text="System Status",
            font=("Arial", 14, "bold"),
            padx=15,
            pady=10
        )

        status_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8
        )

        self.camera_status = self.add_status(
            status_frame,
            "Camera:",
            "STOPPED",
            "red"
        )

        self.hand_status = self.add_status(
            status_frame,
            "Hand Detection:",
            "Not detected",
            "red"
        )

        self.gesture_status = self.add_status(
            status_frame,
            "Recognized Gesture:",
            "UNKNOWN",
            "blue",
            font_size=17
        )

        self.raw_status = self.add_status(
            status_frame,
            "Raw Prediction:",
            "UNKNOWN",
            "black"
        )

        self.mouse_status = self.add_status(
            status_frame,
            "Mouse Control:",
            "DISABLED",
            "red"
        )

        self.action_status = self.add_status(
            status_frame,
            "Current Action:",
            "None",
            "black"
        )

        # ----------------------------------------------
        # Performance monitoring
        # ----------------------------------------------

        performance_frame = tk.LabelFrame(
            status_frame,
            text="Performance Monitoring",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8
        )

        performance_frame.pack(
            fill="x",
            pady=(18, 5)
        )

        self.fps_status = self.add_metric(
            performance_frame,
            "FPS:",
            "0.00"
        )

        self.processing_status = self.add_metric(
            performance_frame,
            "Average Processing Time:",
            "0.00 ms"
        )

        self.detection_rate_status = self.add_metric(
            performance_frame,
            "Hand Detection Rate:",
            "0.00%"
        )

        self.frame_count_status = self.add_metric(
            performance_frame,
            "Total Frames:",
            "0"
        )

        self.session_status = self.add_metric(
            performance_frame,
            "Session Duration:",
            "0.0 seconds"
        )

        # ----------------------------------------------
        # Gesture distribution
        # ----------------------------------------------

        distribution_frame = tk.LabelFrame(
            status_frame,
            text="Gesture Distribution",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8
        )

        distribution_frame.pack(
            fill="x",
            pady=5
        )

        self.gesture_distribution_labels = {}

        gestures = [
            "OPEN PALM",
            "FIST",
            "INDEX ONLY",
            "TWO FINGERS",
            "PINCH",
            "UNKNOWN"
        ]

        for gesture in gestures:

            label = tk.Label(
                distribution_frame,
                text=f"{gesture}: 0",
                font=("Arial", 10),
                anchor="w"
            )

            label.pack(
                fill="x",
                pady=1
            )

            self.gesture_distribution_labels[gesture] = label

        # ----------------------------------------------
        # Application controls
        # ----------------------------------------------

        controls = tk.LabelFrame(
            parent,
            text="Application Controls",
            font=("Arial", 13, "bold"),
            padx=10,
            pady=12
        )

        controls.pack(
            fill="x",
            padx=25,
            pady=12
        )

        self.start_button = tk.Button(
            controls,
            text="START CAMERA",
            font=("Arial", 10, "bold"),
            bg="#2E8B57",
            fg="white",
            command=self.start_camera
        )

        self.start_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.stop_button = tk.Button(
            controls,
            text="STOP CAMERA",
            font=("Arial", 10, "bold"),
            bg="#D9534F",
            fg="white",
            command=self.stop_camera
        )

        self.stop_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.mouse_button = tk.Button(
            controls,
            text="ENABLE MOUSE",
            font=("Arial", 10, "bold"),
            bg="#2874A6",
            fg="white",
            command=self.toggle_mouse
        )

        self.mouse_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.keyboard_button = tk.Button(
            controls,
            text="OPEN KEYBOARD",
            font=("Arial", 10, "bold"),
            command=self.open_keyboard
        )

        self.keyboard_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.emergency_button = tk.Button(
            controls,
            text="EMERGENCY STOP",
            font=("Arial", 10, "bold"),
            bg="#8B0000",
            fg="white",
            command=self.emergency_stop
        )

        self.emergency_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=4
        )

        self.instruction_label = tk.Label(
            parent,
            text="Start the camera, then enable mouse control when ready.",
            font=("Arial", 11),
            fg="gray"
        )

        self.instruction_label.pack(
            pady=(0, 15)
        )

    # --------------------------------------------------
    # 7. Dashboard label helpers
    # --------------------------------------------------

    def add_status(
        self,
        parent,
        title,
        initial_value,
        color,
        font_size=12
    ):

        tk.Label(
            parent,
            text=title,
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w",
            pady=(8, 1)
        )

        label = tk.Label(
            parent,
            text=initial_value,
            font=("Arial", font_size, "bold"),
            fg=color
        )

        label.pack(anchor="w")

        return label

    def add_metric(self, parent, title, initial_value):

        label = tk.Label(
            parent,
            text=f"{title} {initial_value}",
            font=("Arial", 10),
            anchor="w"
        )

        label.pack(
            fill="x",
            pady=2
        )

        return label

    # --------------------------------------------------
    # 8. Reset statistics
    # --------------------------------------------------

    def reset_statistics(self):

        self.total_frames = 0
        self.hand_detected_frames = 0

        self.gesture_counts.clear()

        self.frame_times.clear()
        self.processing_times.clear()

        self.start_time = time.monotonic()
        self.session_saved = False

        self.update_statistics_display()

    # --------------------------------------------------
    # 9. Calculate performance statistics
    # --------------------------------------------------

    def calculate_fps(self):

        if len(self.frame_times) < 2:
            return 0.0

        elapsed = (
            self.frame_times[-1]
            - self.frame_times[0]
        )

        if elapsed <= 0:
            return 0.0

        return (len(self.frame_times) - 1) / elapsed

    def calculate_processing_time(self):

        if not self.processing_times:
            return 0.0

        return (
            sum(self.processing_times)
            / len(self.processing_times)
        ) * 1000

    def calculate_detection_rate(self):

        if self.total_frames == 0:
            return 0.0

        return (
            self.hand_detected_frames
            / self.total_frames
        ) * 100

    # --------------------------------------------------
    # 10. Update performance display
    # --------------------------------------------------

    def update_statistics_display(self):

        fps = self.calculate_fps()

        processing_ms = self.calculate_processing_time()

        detection_rate = self.calculate_detection_rate()

        duration = time.monotonic() - self.start_time

        self.fps_status.config(
            text=f"FPS: {fps:.2f}"
        )

        self.processing_status.config(
            text=f"Average Processing Time: {processing_ms:.2f} ms"
        )

        self.detection_rate_status.config(
            text=f"Hand Detection Rate: {detection_rate:.2f}%"
        )

        self.frame_count_status.config(
            text=f"Total Frames: {self.total_frames}"
        )

        self.session_status.config(
            text=f"Session Duration: {duration:.1f} seconds"
        )

        for gesture, label in self.gesture_distribution_labels.items():

            count = self.gesture_counts.get(gesture, 0)

            label.config(
                text=f"{gesture}: {count}"
            )

    # --------------------------------------------------
    # 11. Save session statistics
    # --------------------------------------------------

    def save_session_statistics(self):

        if self.session_saved or self.total_frames == 0:
            return

        os.makedirs(
            EXPERIMENTS_DIR,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"session_metrics_{timestamp}.csv"

        filepath = os.path.join(
            EXPERIMENTS_DIR,
            filename
        )

        duration = time.monotonic() - self.start_time

        row = {
            "date_time": datetime.now().isoformat(timespec="seconds"),
            "total_frames": self.total_frames,
            "hand_detected_frames": self.hand_detected_frames,
            "hand_detection_rate_percent": round(
                self.calculate_detection_rate(), 2
            ),
            "average_fps_recent_window": round(
                self.calculate_fps(), 2
            ),
            "average_processing_time_ms_recent_window": round(
                self.calculate_processing_time(), 2
            ),
            "session_duration_seconds": round(duration, 2)
        }

        for gesture in [
            "OPEN PALM",
            "FIST",
            "INDEX ONLY",
            "TWO FINGERS",
            "PINCH",
            "UNKNOWN"
        ]:

            safe_name = gesture.lower().replace(" ", "_")

            row[f"{safe_name}_count"] = self.gesture_counts.get(
                gesture,
                0
            )

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=row.keys()
            )

            writer.writeheader()
            writer.writerow(row)

        self.session_saved = True

        print("\nSession statistics saved:")
        print(filepath)

    # --------------------------------------------------
    # 12. Start camera
    # --------------------------------------------------

    def start_camera(self):

        if self.camera_active:
            return

        self.cap = open_camera()

        if self.cap is None:

            self.camera_status.config(
                text="CAMERA ERROR",
                fg="red"
            )

            return

        self.reset_statistics()

        self.gesture_stabilizer.reset()

        self.camera_active = True

        self.camera_status.config(
            text="RUNNING",
            fg="green"
        )

        self.instruction_label.config(
            text="Camera running. Enable mouse control when ready."
        )

        self.process_frame()

    # --------------------------------------------------
    # 13. Stop camera
    # --------------------------------------------------

    def stop_camera(self):

        if not self.camera_active:
            return

        self.camera_active = False
        self.mouse_enabled = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.gesture_stabilizer.reset()

        self.save_session_statistics()

        self.camera_status.config(
            text="STOPPED",
            fg="red"
        )

        self.hand_status.config(
            text="Not detected",
            fg="red"
        )

        self.gesture_status.config(text="UNKNOWN")
        self.raw_status.config(text="UNKNOWN")

        self.mouse_status.config(
            text="DISABLED",
            fg="red"
        )

        self.action_status.config(text="None")

        self.mouse_button.config(
            text="ENABLE MOUSE",
            bg="#2874A6"
        )

        self.camera_label.config(
            image="",
            text="Camera is currently stopped"
        )

        self.camera_label.image = None

        self.update_statistics_display()

        self.instruction_label.config(
            text="Camera stopped. Session statistics saved if frames were processed."
        )

    # --------------------------------------------------
    # 14. Toggle mouse control
    # --------------------------------------------------

    def toggle_mouse(self):

        if not self.camera_active:

            self.instruction_label.config(
                text="Start the camera before enabling mouse control."
            )

            return

        self.mouse_enabled = not self.mouse_enabled

        if self.mouse_enabled:

            self.mouse_status.config(
                text="ENABLED",
                fg="green"
            )

            self.mouse_button.config(
                text="DISABLE MOUSE",
                bg="#D9534F"
            )

            self.instruction_label.config(
                text="Mouse control enabled."
            )

        else:

            self.mouse_status.config(
                text="DISABLED",
                fg="red"
            )

            self.mouse_button.config(
                text="ENABLE MOUSE",
                bg="#2874A6"
            )

            self.action_status.config(text="None")

            self.instruction_label.config(
                text="Mouse control disabled."
            )

    # --------------------------------------------------
    # 15. Emergency stop
    # --------------------------------------------------

    def emergency_stop(self):

        self.mouse_enabled = False

        self.gesture_stabilizer.reset()

        self.mouse_status.config(
            text="EMERGENCY STOP",
            fg="red"
        )

        self.mouse_button.config(
            text="ENABLE MOUSE",
            bg="#2874A6"
        )

        self.gesture_status.config(
            text="UNKNOWN"
        )

        self.action_status.config(
            text="All mouse actions disabled"
        )

        self.mouse_controller.is_clicking = False
        self.mouse_controller.previous_x = None
        self.mouse_controller.previous_y = None
        self.mouse_controller.previous_scroll_y = None

        self.instruction_label.config(
            text="Emergency stop activated. Enable mouse control to resume."
        )

    # --------------------------------------------------
    # 16. Open virtual keyboard
    # --------------------------------------------------

    def open_keyboard(self):

        if (
            self.keyboard_window is not None
            and self.keyboard_window.winfo_exists()
        ):

            self.keyboard_window.lift()
            return

        self.keyboard_window = tk.Toplevel(self.root)

        self.keyboard = VirtualKeyboard(
            self.keyboard_window
        )

        self.keyboard_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_keyboard
        )

    def close_keyboard(self):

        if self.keyboard_window is not None:

            self.keyboard_window.destroy()

        self.keyboard_window = None
        self.keyboard = None

    # --------------------------------------------------
    # 17. Update camera preview
    # --------------------------------------------------

    def update_camera_preview(self, frame):

        display_frame = cv2.resize(
            frame,
            (640, 480)
        )

        success, encoded = cv2.imencode(
            ".ppm",
            display_frame
        )

        if not success:
            return

        photo = tk.PhotoImage(
            data=encoded.tobytes()
        )

        self.camera_label.config(
            image=photo,
            text=""
        )

        self.camera_label.image = photo

    # --------------------------------------------------
    # 18. Update action status
    # --------------------------------------------------

    def update_action_status(self, gesture):

        if not self.mouse_enabled:

            action = "Mouse disabled"

        elif gesture == "INDEX_ONLY":

            action = "Moving cursor"

        elif gesture == "TWO_FINGERS":

            action = "Scrolling"

        elif gesture == "PINCH":

            action = "Left click"

        else:

            action = "Mouse inactive"

        self.action_status.config(
            text=action
        )

    # --------------------------------------------------
    # 19. Process webcam frames
    # --------------------------------------------------

    def process_frame(self):

        if not self.running or not self.camera_active:
            return

        if self.cap is None:
            return

        frame_start = time.perf_counter()

        ret, frame = self.cap.read()

        if not ret:

            self.instruction_label.config(
                text="Could not read camera frame."
            )

            self.stop_camera()
            return

        self.total_frames += 1

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------
        # Hand detected
        # ----------------------------------------------

        if results.hand_landmarks:

            self.hand_detected_frames += 1

            landmarks = results.hand_landmarks[0]

            finger_states, finger_angles = detect_fingers(
                landmarks
            )

            raw_gesture = classify_gesture(
                landmarks,
                finger_states
            )

            gesture = self.gesture_stabilizer.update(
                raw_gesture
            )

            self.gesture_counts[gesture] += 1

            if self.mouse_enabled:

                self.mouse_controller.handle_gesture(
                    gesture,
                    landmarks,
                    finger_states
                )

            else:

                self.mouse_controller.is_clicking = False
                self.mouse_controller.previous_x = None
                self.mouse_controller.previous_y = None
                self.mouse_controller.previous_scroll_y = None

            draw_hand_landmarks(
                frame,
                landmarks
            )

            self.hand_status.config(
                text="Detected",
                fg="green"
            )

            self.gesture_status.config(
                text=gesture
            )

            self.raw_status.config(
                text=raw_gesture
            )

            self.update_action_status(gesture)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Raw: {raw_gesture}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # ----------------------------------------------
        # No hand detected
        # ----------------------------------------------

        else:

            self.gesture_stabilizer.reset()

            self.mouse_controller.is_clicking = False
            self.mouse_controller.previous_x = None
            self.mouse_controller.previous_y = None
            self.mouse_controller.previous_scroll_y = None

            self.gesture_counts["UNKNOWN"] += 1

            self.hand_status.config(
                text="Not detected",
                fg="red"
            )

            self.gesture_status.config(
                text="UNKNOWN"
            )

            self.raw_status.config(
                text="UNKNOWN"
            )

            self.action_status.config(
                text="No hand detected"
            )

            cv2.putText(
                frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------
        # Performance monitoring
        # ----------------------------------------------

        frame_end = time.perf_counter()

        processing_time = frame_end - frame_start

        self.processing_times.append(
            processing_time
        )

        self.frame_times.append(frame_end)

        self.update_statistics_display()

        self.update_camera_preview(frame)

        # Schedule next frame.

        if self.running and self.camera_active:

            self.root.after(
                10,
                self.process_frame
            )

    # --------------------------------------------------
    # 20. Close application
    # --------------------------------------------------

    def close(self):

        if not self.running:
            return

        if self.camera_active:

            self.camera_active = False

            if self.cap is not None:

                self.cap.release()
                self.cap = None

            self.save_session_statistics()

        self.running = False
        self.mouse_enabled = False

        print("Closing Gesture Control System...")

        if self.landmarker is not None:

            self.landmarker.close()
            self.landmarker = None

        if (
            self.keyboard_window is not None
            and self.keyboard_window.winfo_exists()
        ):

            self.keyboard_window.destroy()

        cv2.destroyAllWindows()

        self.root.destroy()

        print("Application closed successfully.")


# --------------------------------------------------
# 21. Main entry point
# --------------------------------------------------

def main():

    root = tk.Tk()

    try:

        app = GestureKeyboardApp(root)

        root.mainloop()

    except Exception as error:

        print("Application error:", error)

        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
