
import time
import pyautogui


class MouseController:

    def __init__(self):

        # Get screen dimensions.
        self.screen_width, self.screen_height = pyautogui.size()

        # Cursor smoothing.
        self.smoothing = 0.25

        self.previous_x = None
        self.previous_y = None

        # Active camera control region.
        # Only the central 80% of the camera frame is used.
        self.margin_x = 0.10
        self.margin_y = 0.10

        # Click protection.
        self.is_clicking = False
        self.last_click_time = 0
        self.click_cooldown = 0.35

        # Scroll tracking.
        self.previous_scroll_y = None

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        print("Mouse controller initialized.")
        print("Screen size:", self.screen_width, self.screen_height)
        print("Active control region: Central 80%")

    # --------------------------------------------------
    # 1. Map camera coordinates to screen coordinates
    # --------------------------------------------------

    def map_coordinates(self, x, y):

        # Map the active camera region to the full screen.
        mapped_x = (x - self.margin_x) / (1 - 2 * self.margin_x)
        mapped_y = (y - self.margin_y) / (1 - 2 * self.margin_y)

        # Clamp values between 0 and 1.
        mapped_x = max(0, min(mapped_x, 1))
        mapped_y = max(0, min(mapped_y, 1))

        # Convert to screen coordinates.
        screen_x = int(mapped_x * (self.screen_width - 1))
        screen_y = int(mapped_y * (self.screen_height - 1))

        return screen_x, screen_y

    # --------------------------------------------------
    # 2. Move cursor
    # --------------------------------------------------

    def move_cursor(self, landmarks):

        index_tip = landmarks[8]

        # Convert camera coordinates to screen coordinates.
        target_x, target_y = self.map_coordinates(
            index_tip.x,
            index_tip.y
        )

        # Initialize cursor position.
        if self.previous_x is None:

            self.previous_x = target_x
            self.previous_y = target_y

        # Apply exponential smoothing.
        smooth_x = (
            self.smoothing * target_x
            + (1 - self.smoothing) * self.previous_x
        )

        smooth_y = (
            self.smoothing * target_y
            + (1 - self.smoothing) * self.previous_y
        )

        self.previous_x = smooth_x
        self.previous_y = smooth_y

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

    # --------------------------------------------------
    # 3. Left click
    # --------------------------------------------------

    def click(self):

        current_time = time.monotonic()

        if current_time - self.last_click_time < self.click_cooldown:
            return

        pyautogui.click(button="left")

        self.last_click_time = current_time

        print("Left click!")

    # --------------------------------------------------
    # 4. Scroll
    # --------------------------------------------------

    def scroll(self, landmarks):

        index_y = landmarks[8].y
        middle_y = landmarks[12].y

        current_y = (index_y + middle_y) / 2

        if self.previous_scroll_y is not None:

            movement = current_y - self.previous_scroll_y

            if abs(movement) > 0.015:

                scroll_amount = int(-movement * 500)

                scroll_amount = max(
                    -8,
                    min(scroll_amount, 8)
                )

                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)

        self.previous_scroll_y = current_y

    # --------------------------------------------------
    # 5. Handle gestures
    # --------------------------------------------------

    def handle_gesture(self, gesture, landmarks, finger_states=None):

        gesture = str(gesture).upper().replace(" ", "_")

        # PINCH: Left click.
        if gesture == "PINCH":

            if not self.is_clicking:
                self.click()
                self.is_clicking = True

            return

        else:
            self.is_clicking = False

        # TWO FINGERS: Scroll.
        if gesture in ["TWO_FINGERS", "SCROLL", "PEACE"]:

            self.scroll(landmarks)

            self.previous_x = None
            self.previous_y = None

            return

        # INDEX ONLY: Move cursor.
        if gesture in [
            "INDEX_ONLY",
            "MOVE_CURSOR",
            "ONE_FINGER",
            "INDEX",
            "ONE"
        ]:

            self.move_cursor(landmarks)

            self.previous_scroll_y = None

            return

        # Other gestures: Pause mouse actions.
        self.previous_x = None
        self.previous_y = None
        self.previous_scroll_y = None
