import numpy as np
import random
import tkinter as tk
from tkinter import ttk, Canvas, Button, Frame, Label
import threading
from queue import Queue

# 6x6 Grid Definition
grid_text = [
    "S . . W . .",
    ". W . . . .",
    ". . W W . O",
    ". O . W . O",
    ". W . O . C",
    ". . W . . .",
]


class QLearningTrainer(threading.Thread):
    def __init__(self, update_queue, episodes=5000):
        super().__init__()
        self.update_queue = update_queue
        self.episodes = episodes
        self.running = True
        self.alpha = 0.2  # Learning rate, controls how much new info affects old info
        self.gamma = 0.95  # Discount factor, determines importance of future rewards
        self.start_epsilon = 0.6  # Initial exploration rate

    def run(self):
        Q = np.zeros((*grid_size, len(actions)))  # Q-table initialization
        epsilon = self.start_epsilon  # Set initial exploration rate

        for episode in range(self.episodes):
            if not self.running:
                break

            position = start_pos  # Start at initial position
            total_reward = 0
            epsilon = max(0.05, epsilon * 0.997)  # Decrease epsilon over time

            while True:
                if random.random() < epsilon or np.max(Q[position]) == 0:
                    action = random.choice(actions)  # Explore or pick random action
                else:
                    action = actions[np.argmax(Q[position])]  # Choose best action

                next_pos, is_wall = get_next_position(position, action)

                # Calculate reward
                if next_pos == cheese_pos:
                    reward = 10  # Mouse found the cheese, big reward
                    episode_end = True
                elif next_pos in holes:
                    reward = -5  # Mouse fell into hole, big penalty
                    episode_end = True
                elif is_wall:
                    reward = -2  # Hitting a wall gives penalty
                    episode_end = False
                else:
                    reward = rewards[next_pos]  # Default rewards for other cells
                    episode_end = False

                # Update Q-value using Q-learning formula
                old_value = Q[position][actions.index(action)]
                next_max = 0 if episode_end else np.max(Q[next_pos])
                new_value = old_value + self.alpha * (
                    reward + self.gamma * next_max - old_value
                )
                Q[position][actions.index(action)] = new_value

                total_reward += reward
                if episode_end or next_pos == position:
                    break
                position = next_pos

            if episode % 50 == 0:
                self.update_queue.put(("progress", episode, epsilon, total_reward))

        self.update_queue.put(("done", Q))  # Training complete, send final Q-table

    def stop(self):
        self.running = False  # Stop training


def parse_grid(grid_text):
    start = None
    cheese = None
    holes = []
    walls = []
    grid_size = (len(grid_text), len(grid_text[0].split()))
    rewards = np.zeros(grid_size)
    cell_types = np.empty(grid_size, dtype="object")

    for x, row in enumerate(grid_text):
        for y, cell in enumerate(row.split()):
            cell_types[x, y] = cell
            if cell == "S":
                start = (x, y)  # Mark start position
                rewards[x, y] = -0.1  # Small penalty for each move
            elif cell == "C":
                cheese = (x, y)  # Goal position
                rewards[x, y] = 10  # Big reward for reaching cheese
            elif cell == "O":
                holes.append((x, y))  # Trap positions
                rewards[x, y] = -5  # Penalty for falling into a hole
            elif cell == "W":
                walls.append((x, y))  # Wall positions
                rewards[x, y] = -2  # Small penalty for hitting a wall
            else:
                rewards[x, y] = -0.3  # Small penalty for each move

    return start, cheese, holes, walls, rewards, cell_types


start_pos, cheese_pos, holes, walls, rewards, cell_types = parse_grid(grid_text)
grid_size = cell_types.shape
actions = ["up", "down", "left", "right"]

# GUI Setup
cell_size = 60
root = tk.Tk()
root.title("Q-Maze Trainer")
update_queue = Queue()

# Training Controls Frame
control_frame = Frame(root)
control_frame.pack(pady=10)

train_btn = Button(
    control_frame, text="Start Training", command=lambda: start_training()
)
train_btn.pack(side="left", padx=5)

progress_bar = ttk.Progressbar(
    control_frame, orient="horizontal", length=200, mode="determinate"
)
progress_bar.pack(side="left", padx=5)

status_frame = Frame(root)
status_frame.pack(pady=5)

episode_label = Label(status_frame, text="Episode: 0/5000")
episode_label.pack(side="left", padx=10)
epsilon_label = Label(status_frame, text="Epsilon: 0.600")
epsilon_label.pack(side="left", padx=10)
reward_label = Label(status_frame, text="Latest Reward: 0.00")
reward_label.pack(side="left", padx=10)

# Maze Visualization
canvas = Canvas(root, width=360, height=360)
canvas.pack()

# Create grid
colors = {"W": "gray30", "O": "firebrick", "C": "gold", "S": "dodgerblue"}
for x in range(6):
    for y in range(6):
        cell = cell_types[x, y]
        color = colors.get(cell, "white smoke")
        canvas.create_rectangle(
            y * cell_size,
            x * cell_size,
            (y + 1) * cell_size,
            (x + 1) * cell_size,
            fill=color,
            outline="black",
        )
