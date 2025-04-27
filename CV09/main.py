import numpy as np
import matplotlib.pyplot as plt


def midpoint_displacement(n_segments, roughness, iterations):

    x = np.array([0.0, 1.0])
    y = np.array([0.5, 0.5])

    for _ in range(iterations):
        new_x, new_y = [], []
        # projdeme kazdou dvojici sousednich bodu
        for i in range(len(x) - 1):
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            # najdeme stred useku
            xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            # pricti nahodny posun podle roughness
            displacement = (np.random.rand() - 0.5) * roughness
            # uloz puvodni bod a novy stred (s nahodnym posunem ve vysce)
            new_x.extend([x1, xm])
            new_y.extend([y1, ym + displacement])

        new_x.append(x[-1])
        new_y.append(y[-1])
        # dalsi iterace bude mit dvojnasobny pocet bodu
        x, y = np.array(new_x), np.array(new_y)
        # zmensi roughness kazdou iteraci na polovinu
        roughness *= 0.5

    # nakonec potrebujem rovnou hustotu x hodnot, proto interpolace
    xi = np.linspace(0, 1, n_segments)
    yi = np.interp(xi, x, y)
    return xi, yi


def normalize_and_shift(h, shift):
    h = (h - h.min()) / (h.max() - h.min()) * 0.25
    return h + shift


def main():
    n = 1024
    x_p, h_p = midpoint_displacement(n, roughness=0.02, iterations=2)
    x_g, h_g = midpoint_displacement(n, roughness=0.05, iterations=4)
    x_h, h_h = midpoint_displacement(n, roughness=0.1, iterations=15)
    x_m, h_m = midpoint_displacement(n, roughness=0.4, iterations=10)

    h_p = normalize_and_shift(h_p, 0.0)
    h_g = normalize_and_shift(h_g, 0.25)
    h_h = normalize_and_shift(h_h, 0.50)
    h_m = normalize_and_shift(h_m, 0.75)

    # vykresleni
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_between(x_p, 0, h_p, color="gold", alpha=1, label="Plains")
    ax.fill_between(x_g, h_p, h_g, color="lightgreen", alpha=1, label="Grasslands")
    ax.fill_between(x_h, h_g, h_h, color="darkseagreen", alpha=1, label="Smooth Hills")
    ax.fill_between(
        x_m, h_h, h_m, color="saddlebrown", alpha=1, label="Rough Mountains"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
