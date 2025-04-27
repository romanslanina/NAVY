import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Ovladani
# levym tlacitkem mysi -zoom in
# b- zoom out
# r- reset
# j- julia
# m- mandelbrot


class FractalZoom:
    def __init__(self):
        # zakladni hodnoty
        self.max_iter = 100  # kolik iteraci max pouzijeme
        self.zoom_factor = 0.5  # o kolik zmensime oblast pri zoomu
        self.history = []  # ulozene predchozi pohledy pro zoom out

        choice = (
            input("Stiskni 'j' pro Julia nebo 'm' pro Mandelbrot: ").strip().lower()
        )
        self.fractal_type = "Julia" if choice == "j" else "Mandelbrot"
        self.julia_c = complex(-0.5, 0.58)  # konstanta pro Julia

        self.reset_ranges()  # nastavime pocatecni oblast

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        Button(ax_reset, "Reset").on_clicked(
            lambda event: self.on_key(type("E", (), {"key": "r"})())
        )

        self.update_plot()

    def reset_ranges(self):
        # rozsah podle fraktalu
        if self.fractal_type == "Mandelbrot":
            self.xmin, self.xmax = -2.0, 1.0
            self.ymin, self.ymax = -1.0, 1.0
        else:
            self.xmin, self.xmax = -1.5, 1.5
            self.ymin, self.ymax = -1.5, 1.5

    def mandelbrot(self, w, h):
        # vytvorime matici s hodnotami iteraci pro Mandelbrotovu mnozinu
        data = [[0] * w for _ in range(h)]
        # Pro kazdy pixel vygenerujeme bod na komplexni rovine
        for i in range(w):
            for j in range(h):
                # Prevedeni indexu (i, j) na komplexni cislo c
                x = self.xmin + (self.xmax - self.xmin) * i / w
                y = self.ymin + (self.ymax - self.ymin) * j / h
                c = complex(x, y)
                z = 0  # pocatecni z
                count = 0
                while abs(z) <= 2 and count < self.max_iter:
                    z = z * z + c
                    count += 1
                # ulozime pocet kroku (barva pixelu)
                data[j][i] = count
        return data

    def julia(self, w, h):
        # podobne pro Julia fraktal, jen pridame konstantu julia_c
        data = [[0] * w for _ in range(h)]
        for i in range(w):
            for j in range(h):
                x = self.xmin + (self.xmax - self.xmin) * i / w
                y = self.ymin + (self.ymax - self.ymin) * j / h
                z = complex(x, y)
                count = 0
                while abs(z) <= 2 and count < self.max_iter:
                    z = z * z + self.julia_c
                    count += 1
                data[j][i] = count
        return data

    def update_plot(self):
        # prekresleni fraktalu podle aktualnich rozsahu
        w, h = 400, 400
        data = (
            self.mandelbrot(w, h)
            if self.fractal_type == "Mandelbrot"
            else self.julia(w, h)
        )

        self.ax.clear()
        self.ax.imshow(
            data,
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            origin="lower",
            cmap="hsv",
            vmin=0,
            vmax=self.max_iter,
        )
        title = f"{self.fractal_type}"
        if self.fractal_type == "Julia":
            title += f" (c={self.julia_c})"
        self.ax.set_title(title)
        self.fig.canvas.draw()

    def on_click(self, event):
        # na levy klik zoom in
        if event.inaxes != self.ax:
            return
        self.history.append((self.xmin, self.xmax, self.ymin, self.ymax))
        cx, cy = event.xdata, event.ydata
        dx = (self.xmax - self.xmin) * self.zoom_factor / 2
        dy = (self.ymax - self.ymin) * self.zoom_factor / 2
        self.xmin, self.xmax = cx - dx, cx + dx
        self.ymin, self.ymax = cy - dy, cy + dy
        self.update_plot()

    def on_key(self, event):
        key = event.key
        if key == "b" and self.history:
            # zpet k predchozimu pohledu
            self.xmin, self.xmax, self.ymin, self.ymax = self.history.pop()
            self.update_plot()
        elif key == "r":
            # vrat na vychozi
            self.history.clear()
            self.reset_ranges()
            self.update_plot()
        elif key in ("j", "m"):
            # prepni typ fraktalu a zacni znova
            self.fractal_type = "Julia" if key == "j" else "Mandelbrot"
            self.history.clear()
            self.reset_ranges()
            self.update_plot()


if __name__ == "__main__":
    z = FractalZoom()
    plt.show()
