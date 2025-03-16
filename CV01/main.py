import numpy as np
import matplotlib.pyplot as plt

# Generate random points in range
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
y = np.random.uniform(-10, 40, 100)


# True boundary line
def reference_line(x):
    return 3 * x + 2


# Create binary labels below = -1, above = 1
labels = np.sign(y - reference_line(x))
labels[labels == 0] = 1  # on the line => 1

# Prepare data matrix and labels
data = np.column_stack((x, y))
targets = labels


class SimplePerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0  # without initial bias

        # Update weights through multiple passes
        for _ in range(self.epochs):
            for idx in range(n_samples):
                prediction = np.dot(X[idx], self.weights) + self.bias
                if y[idx] * prediction <= 0:  # Misclassification check
                    update = self.lr * y[idx]
                    self.weights += update * X[idx]
                    self.bias += update

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)


# Create and train perceptron
classifier = SimplePerceptron()
classifier.train(data, targets)
results = classifier.predict(data)

# visualisation
plt.figure(figsize=(8, 6))
x_vals = np.linspace(-10, 10, 100)
plt.plot(x_vals, reference_line(x_vals), "k--", label="Reference Line")

for class_val, color in [(-1, "red"), (1, "blue")]:
    mask = results == class_val
    plt.scatter(x[mask], y[mask], c=color, label=f"Class {class_val}")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Perceptron Classification Results")
plt.legend()
plt.grid(True)
plt.show()
