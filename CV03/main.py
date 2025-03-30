import numpy as np
import tkinter as tk
from tkinter import messagebox, Canvas, Button, Label, Frame


# Simple Hopfield Network for pattern recognition
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size  # Size of the network (we're using a 5x5 grid, so 25 neurons)
        self.weights = np.zeros(
            (size, size)
        )  # Initialize weights (connections between neurons)
        self.trained_patterns = []  # Keep track of trained patterns

    def train(self, pattern):
        self.trained_patterns.append(pattern.copy())  # Save the pattern
        self.weights.fill(0)  # Reset weights before retraining
        for p in self.trained_patterns:  # Go through each saved pattern
            p = p.reshape(self.size, 1)  # Convert to column vector
            self.weights += np.dot(p, p.T)  # Update weights (basic learning rule)
        np.fill_diagonal(
            self.weights, 0
        )  # Make sure neurons don’t reinforce themselves

    def recover(self, pattern, synchronous=True, max_iter=10):
        recovered = pattern.copy()
        for _ in range(max_iter):  # Give it a few tries to settle
            if synchronous:
                recovered = np.sign(
                    self.weights @ recovered
                )  # Update everything at once
            else:
                for i in range(self.size):  # Or update one at a time (async mode)
                    recovered[i] = np.sign(np.dot(self.weights[i], recovered))
            if np.array_equal(recovered, pattern):  # If it stops changing, we're done
                break
        return recovered


# Draws a pattern on the canvas (black = active, white = inactive)
def draw_pattern(canvas, pattern, size, cell_size=40):
    canvas.delete("all")  # Clear previous drawing
    for i in range(size):
        x, y = (i % 5) * cell_size, (
            i // 5
        ) * cell_size  # Figure out where to draw each square
        color = "black" if pattern[i] == 1 else "white"
        canvas.create_rectangle(
            x, y, x + cell_size, y + cell_size, fill=color, outline="gray"
        )


# Handles clicking on a square to flip its state (black <-> white)
def toggle_cell(event):
    global user_pattern
    cell_size = 40
    row, col = (
        event.y // cell_size,
        event.x // cell_size,
    )  # Convert click position to grid index
    index = row * 5 + col  # Get the right index for our 1D array
    if 0 <= index < 25:
        user_pattern[index] *= -1  # Flip the value
        draw_pattern(pattern_canvas, user_pattern, 25)


# Train the network with whatever the user has drawn
def on_train():
    global network, user_pattern, saved_patterns_frame
    network.train(user_pattern)
    update_saved_patterns()  # Refresh saved patterns


# Try to recover a pattern from noise or distortion
def on_recover(synchronous=True):
    global network, user_pattern
    recovered_pattern = network.recover(user_pattern, synchronous)
    draw_pattern(
        pattern_canvas, recovered_pattern, 25
    )  # Show what the network thinks it should be


# Resets the grid to a blank slate
def reset_pattern():
    global user_pattern
    user_pattern = np.ones(25) * -1
    draw_pattern(pattern_canvas, user_pattern, 25)


# Show all the patterns the network has learned
def update_saved_patterns():
    for widget in saved_patterns_frame.winfo_children():
        widget.destroy()  # Clear previous patterns
    for pattern in network.trained_patterns:
        canvas = Canvas(
            saved_patterns_frame,
            width=100,
            height=100,
            bg="white",
            highlightthickness=1,
            relief="solid",
        )
        canvas.pack(pady=5)
        draw_pattern(
            canvas, pattern, 25, cell_size=20
        )  # Draw a mini version of each trained pattern


# GUI Setup
root = tk.Tk()
root.title("Hopfield Network GUI")
root.configure(bg="#f4f4f4")
network = HopfieldNetwork(25)  # Create the network (5x5 grid)
user_pattern = np.ones(25) * -1

main_frame = Frame(root, bg="#f4f4f4")
main_frame.pack(pady=10)

# The main canvas where users draw their patterns
pattern_canvas = Canvas(
    main_frame, width=200, height=200, bg="white", highlightthickness=2, relief="ridge"
)
pattern_canvas.grid(row=0, column=0, padx=10, pady=10)
pattern_canvas.bind("<Button-1>", toggle_cell)
draw_pattern(pattern_canvas, user_pattern, 25)

controls_frame = Frame(main_frame, bg="#f4f4f4")
controls_frame.grid(row=0, column=1, padx=10)

# Buttons for interacting with the network
btn_train = Button(
    controls_frame,
    text="Train",
    command=on_train,
    bg="#007acc",  # Blue
    fg="white",
    font=("Arial", 12),
    width=15,
)
btn_train.pack(pady=5)

btn_recover_sync = Button(
    controls_frame,
    text="Recover (Sync)",
    command=lambda: on_recover(True),
    bg="#28a745",  # Green
    fg="white",
    font=("Arial", 12),
    width=15,
)
btn_recover_sync.pack(pady=5)

btn_recover_async = Button(
    controls_frame,
    text="Recover (Async)",
    command=lambda: on_recover(False),
    bg="#ff9800",  # Orange
    fg="white",
    font=("Arial", 12),
    width=15,
)
btn_recover_async.pack(pady=5)

btn_reset = Button(
    controls_frame,
    text="Reset",
    command=reset_pattern,
    bg="#dc3545",  # Red
    fg="white",
    font=("Arial", 12),
    width=15,
)
btn_reset.pack(pady=5)

# Section for showing saved patterns
Label(root, text="Saved Patterns", font=("Arial", 14), bg="#f4f4f4").pack()
saved_patterns_frame = Frame(root, bg="#f4f4f4")
saved_patterns_frame.pack()

root.mainloop()  # Run the app
