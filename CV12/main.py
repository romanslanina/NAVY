import numpy as np, matplotlib.pyplot as plt, tkinter as tk
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# grid size + cell codes
H, W = 150, 150
EMPTY, TREE, FIRE = 0, 1, 2
rng = np.random.default_rng()


# build initial forest -------------------------------------------------
def init_grid(d):
    g = (rng.random((H, W)) < d).astype(np.uint8)  # fill with trees (prob = d)
    g[rng.random((H, W)) < 0.001] = FIRE  #  spark
    return g


# one CA step ----------------------------------------------------------
def step(g, p, f):
    n = (
        np.roll(g == FIRE, 1, 0)
        | np.roll(g == FIRE, -1, 0)  # any neighbour on fire?
        | np.roll(g == FIRE, 1, 1)
        | np.roll(g == FIRE, -1, 1)
    )
    new = g.copy()
    new[g == FIRE] = EMPTY  # burned → empty
    ignite = ((g == TREE) & n) | ((g == TREE) & (rng.random(g.shape) < f))
    new[ignite] = FIRE  # tree catches
    grow = (g == EMPTY) & (rng.random(g.shape) < p)
    new[grow] = TREE  # empty → tree
    return new


root = tk.Tk()
root.title("Forest‑Fire")
root.geometry("1200x700")
panel = tk.Frame(root)
panel.pack(side="left", fill="y", padx=8)


# helper to add a labeled slider
def slider(lbl, a, b, init, step):
    tk.Label(panel, text=lbl).pack(anchor="w")
    s = tk.Scale(panel, from_=a, to=b, resolution=step, orient="horizontal", length=220)
    s.set(init)
    s.pack()
    return s


# controls
p_s = slider("growth p", 0.0, 0.10, 0.05, 0.005)
f_s = slider("lightning f", 0.0, 0.02, 0.001, 0.0005)
d_s = slider("init density", 0.1, 1.0, 0.6, 0.05)
spd_s = slider("delay ms", 10, 200, 60, 10)

paused = tk.BooleanVar(False)


def pause():
    paused.set(not paused.get())
    btn_p["text"] = ("Resume", "Pause")[not paused.get()]


def reset():
    global grid
    grid = init_grid(d_s.get())


btn_p = tk.Button(panel, text="Pause", command=pause)
btn_p.pack(pady=4, fill="x")
tk.Button(panel, text="Reset", command=reset).pack(fill="x")

# Matplotlib view ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")
fire_cmap = ListedColormap(["black", "#006400", "#ff3300"])  # empty / tree / fire
grid = init_grid(d_s.get())
img = ax.imshow(grid, cmap=fire_cmap, vmin=0, vmax=2, interpolation="nearest")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side="right", expand=1)


# animation ------------------------------------------------------------
def update(_):
    global grid
    if not paused.get():
        grid = step(grid, p_s.get(), f_s.get())
        img.set_data(grid)
    return (img,)


def refresh(*_):
    ani.event_source.interval = spd_s.get()


spd_s.configure(command=refresh)

ani = FuncAnimation(fig, update, interval=spd_s.get(), blit=True)
root.mainloop()
