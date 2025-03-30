import gymnasium as gym
import numpy as np
import random
import tkinter as tk
from tkinter import ttk, Canvas, Button, Frame, Label
import threading
from queue import Queue

# State binning setup - 10 bins per dimension seems to work okay
BINS = [
    np.linspace(-2.4, 2.4, 10),  # Cart position
    np.linspace(-4.0, 4.0, 10),  # Velocity range
    np.linspace(-0.25, 0.25, 10),  # Angle with buffer
    np.linspace(-4.0, 4.0, 10),  # Angular velocity
]


class QLearningTrainer(threading.Thread):
    def __init__(self, update_queue, episodes=10000):
        super().__init__()
        self.update_queue = update_queue  # For talking to GUI
        self.episodes = episodes
        self.running = True
        self.env = gym.make("CartPole-v1")

        # Initialize Q-table with zeros
        self.q_table = np.zeros(
            (
                len(BINS[0]),
                len(BINS[1]),
                len(BINS[2]),
                len(BINS[3]),
                self.env.action_space.n,
            )
        )

        # These values worked better than initial tries
        self.alpha = 0.1  # How much to learn from new stuff
        self.gamma = 0.99  # Importance of future rewards
        self.epsilon = 1.0  # Start with random actions
        self.epsilon_decay = 0.9995  # Reduce randomness over time

    def discretize(self, state):
        # Squeeze continuous states into our bins
        indices = []
        for i in range(4):
            clipped = np.clip(state[i], BINS[i][0], BINS[i][-1])
            indices.append(np.digitize(clipped, BINS[i]) - 1)
        return tuple(indices)

    def run(self):
        # Main training loop
        for episode in range(self.episodes):
            if not self.running:
                break

            state = self.discretize(self.env.reset()[0])
            total_reward = 0
            done = False
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            while not done:
                # Sometimes pick random action to explore
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                obs, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(obs)

                # Update Q-value using Bellman equation
                old = self.q_table[state][action]
                future = np.max(self.q_table[next_state])
                new = (1 - self.alpha) * old + self.alpha * (
                    reward + self.gamma * future
                )
                self.q_table[state][action] = new

                state = next_state
                total_reward += reward

            # Send progress update to GUI
            self.update_queue.put(("progress", episode, self.epsilon, total_reward))

        self.update_queue.put(("done", self.q_table))
        self.env.close()

    def stop(self):
        self.running = False


# GUI stuff below - basic Tkinter setup
root = tk.Tk()
root.title("CartPole Q-Learning Trainer")
update_queue = Queue()

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

episode_label = Label(status_frame, text="Episode: 0/10000")
episode_label.pack(side="left", padx=10)
epsilon_label = Label(status_frame, text="Epsilon: 1.000")
epsilon_label.pack(side="left", padx=10)
reward_label = Label(status_frame, text="Latest Reward: 0.00")
reward_label.pack(side="left", padx=10)

# Visualization setup
canvas_width = 600
canvas_height = 200
canvas = Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

cart_width = 40
cart_height = 30
pole_length = 80
cart_y = canvas_height // 2 + 20

track = canvas.create_rectangle(0, cart_y - 5, canvas_width, cart_y + 5, fill="gray")
cart = canvas.create_rectangle(0, 0, cart_width, cart_height, fill="blue")
pole = canvas.create_line(0, 0, 0, -pole_length, width=4, fill="red")


def update_visualization(x, theta):
    # Update cart and pole position on canvas
    cart_x = (x + 2.4) * (canvas_width - cart_width) / 4.8
    canvas.coords(
        cart,
        cart_x,
        cart_y - cart_height // 2,
        cart_x + cart_width,
        cart_y + cart_height // 2,
    )

    pole_top_x = cart_x + cart_width // 2 + pole_length * np.sin(theta)
    pole_top_y = cart_y - cart_height // 2 - pole_length * np.cos(theta)
    canvas.coords(
        pole,
        cart_x + cart_width // 2,
        cart_y - cart_height // 2,
        pole_top_x,
        pole_top_y,
    )


def start_training():
    global trainer
    train_btn.config(state=tk.DISABLED)
    progress_bar["value"] = 0
    trainer = QLearningTrainer(update_queue)
    trainer.start()
    root.after(100, process_updates)


def process_updates():
    # Handle progress updates from training thread
    try:
        while True:
            data = update_queue.get_nowait()
            if data[0] == "progress":
                _, episode, epsilon, reward = data
                progress = (episode / 10000) * 100
                progress_bar["value"] = progress
                episode_label.config(text=f"Episode: {episode}/10000")
                epsilon_label.config(text=f"Epsilon: {epsilon:.3f}")
                reward_label.config(text=f"Latest Reward: {reward:.2f}")
            elif data[0] == "done":
                train_btn.config(state=tk.NORMAL)
    except:
        pass
    root.after(100, process_updates)


def run_simulation():
    # Test trained policy
    env = gym.make("CartPole-v1")

    def simulate():
        state = env.reset()[0]
        done = False

        while not done:
            discretized = trainer.discretize(state)
            action = np.argmax(trainer.q_table[discretized])
            print(
                f"Action: {'Left' if action == 0 else 'Right'} | State: {np.round(state, 2)}"
            )

            state, _, done, _, _ = env.step(action)

            if done:
                # Show why it failed
                if abs(state[0]) > 2.4:
                    print("FAILED: Cart went too far!")
                elif abs(state[2]) > 0.2095:
                    print("FAILED: Pole fell over!")
                else:
                    print("FAILED: Ran out of time!")

            update_visualization(state[0], state[2])
            root.update()
            root.after(50)

    simulate()


Button(root, text="Run Simulation", command=run_simulation).pack(pady=10)

root.mainloop()
