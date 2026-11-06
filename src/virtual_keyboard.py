
import tkinter as tk


class VirtualKeyboard:

    def __init__(self, root):

        self.root = root

        self.root.title("Gesture-Controlled Virtual Keyboard")
        self.root.geometry("1000x500")
        self.root.minsize(800, 400)

        self.root.attributes("-topmost", True)

        # Heading
        heading = tk.Label(
            root,
            text="Gesture-Controlled Virtual Keyboard",
            font=("Arial", 20, "bold")
        )
        heading.pack(pady=10)

        # Text area
        self.text_area = tk.Text(
            root,
            height=5,
            font=("Arial", 16),
            wrap="word"
        )

        self.text_area.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # Keyboard container
        self.keyboard_frame = tk.Frame(root)

        self.keyboard_frame.pack(
            expand=True,
            fill="both",
            padx=15,
            pady=10
        )

        self.keyboard_layout = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M"],
            ["SPACE", "BACKSPACE", "ENTER", "CLEAR"]
        ]

        self.create_keyboard()

    # --------------------------------------------------
    # Create keyboard buttons
    # --------------------------------------------------

    def create_keyboard(self):

        for row_index, row in enumerate(self.keyboard_layout):

            row_frame = tk.Frame(self.keyboard_frame)

            row_frame.pack(
                expand=True,
                fill="both",
                pady=3
            )

            for key in row:

                button = tk.Button(
                    row_frame,
                    text=key,
                    font=("Arial", 13, "bold"),
                    command=lambda value=key: self.press_key(value),
                    relief="raised",
                    bd=2
                )

                button.pack(
                    side="left",
                    expand=True,
                    fill="both",
                    padx=3,
                    pady=3
                )

    # --------------------------------------------------
    # Handle key presses
    # --------------------------------------------------

    def press_key(self, key):

        if key == "SPACE":

            self.text_area.insert("insert", " ")

        elif key == "BACKSPACE":

            current_position = self.text_area.index("insert")

            if current_position != "1.0":
                self.text_area.delete("insert-1c", "insert")

        elif key == "ENTER":

            self.text_area.insert("insert", "\n")

        elif key == "CLEAR":

            self.text_area.delete("1.0", "end")

        else:

            self.text_area.insert("insert", key)


def main():

    root = tk.Tk()

    keyboard = VirtualKeyboard(root)

    root.mainloop()


if __name__ == "__main__":
    main()
