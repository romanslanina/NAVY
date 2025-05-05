import random
import matplotlib.pyplot as plt


# Function for afinite transformation based on task
def transform(point, params):

    x, y, z = point
    a, b, c, d, e, f, g, h, i, j, k, l = params
    x_new = a * x + b * y + c * z + j
    y_new = d * x + e * y + f * z + k
    z_new = g * x + h * y + i * z + l
    return (x_new, y_new, z_new)


# Params
first_model = [
    (0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00),
    (0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00),
    (-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00),
    (0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00),
]


second_model = [
    (0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00),
    (0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00),
    (-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00),
    (0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00),
]


def generate_ifs(model, iterations=100000, initial_point=(0, 0, 0)):
    current_point = initial_point
    # start with inital point
    points = [current_point]

    # iterate and randomly transform the point based on the model
    for _ in range(iterations):
        params = random.choice(model)  # equal probability
        current_point = transform(current_point, params)
        points.append(current_point)
    return points


def plot_fractal_3d(points, title="IFS Fractal"):
    # extract coords and plot ecah point
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    z_vals = [p[2] for p in points]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_vals, y_vals, z_vals, s=5, c="green", marker="o")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def main():
    # Pick model
    print("Select the IFS fractal model:")
    print("1: First Model")
    print("2: Second Model")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        model = first_model
        title = "First Model"
    elif choice == "2":
        model = second_model
        title = "Second Model"
    else:
        print("Invalid choice. Defaulting to first model.")
        model = first_model
        title = "First Model"

    iterations = 10000  # iterations in generation - impacts detail
    points = generate_ifs(model, iterations=iterations)

    plot_fractal_3d(points, title)


if __name__ == "__main__":
    main()
