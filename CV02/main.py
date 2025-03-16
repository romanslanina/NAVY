import numpy as np
import matplotlib.pyplot as plt


# squash numbers between 0-1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# how to adjust weights
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Set up connections between layers with random starting values
        self.learning_rate = learning_rate
        self.weights1 = np.random.rand(input_size, hidden_size) - 0.5
        self.weights2 = np.random.rand(hidden_size, output_size) - 0.5
        self.bias1 = np.random.rand(hidden_size) - 0.5
        self.bias2 = np.random.rand(output_size) - 0.5
        self.loss_history = []  # Memory for tracking progress

    # flow of information from input to output
    def think(self, X):
        self.hidden_layer_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights2) + self.bias2
        )
        final_output = sigmoid(output_layer_input)
        return final_output

    # adjust weights based on errors
    def learn_from_mistakes(self, X, y, output):
        # Calculate error at output layer
        output_error = y - output
        output_adjustment = output_error * sigmoid_derivative(output)

        # figure out hidden layer responsibility for error
        hidden_error = output_adjustment.dot(self.weights2.T)
        hidden_adjustment = hidden_error * sigmoid_derivative(self.hidden_layer_output)
        # Update output layer connections
        self.weights2 += (
            self.hidden_layer_output.T.dot(output_adjustment) * self.learning_rate
        )
        self.bias2 += np.sum(output_adjustment, axis=0) * self.learning_rate
        # update hidden layer connections
        self.weights1 += X.T.dot(hidden_adjustment) * self.learning_rate
        self.bias1 += np.sum(hidden_adjustment, axis=0) * self.learning_rate

    def train(self, X, y, train_rounds=10000):  # training loop
        for round in range(train_rounds):
            guess = self.think(X)
            self.learn_from_mistakes(X, y, guess)
            error = np.mean(np.abs(y - guess))
            self.loss_history.append(error)
            if round % 1000 == 0:
                print(f"train round {round}: Current error {error:.4f}")

    def show_learning(self):  # visualize error decrese (hopefully)
        plt.plot(self.loss_history)
        plt.xlabel("train Rounds")
        plt.ylabel("Mistakes Made")
        plt.title("Learning Progress")
        plt.show()

    def show_decision_making(self, X, y):
        # create grid of test points covering all possibilities
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        # get all predictions
        grid_predictions = self.think(np.c_[xx.ravel(), yy.ravel()])
        grid_predictions = grid_predictions.reshape(xx.shape)
        plt.contourf(
            xx, yy, grid_predictions, levels=[0, 0.5, 1], alpha=0.5, cmap="coolwarm"
        )
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors="k", cmap="coolwarm")
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.title("Decision Making Map")
        plt.show()


training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
correct_answers = np.array([[0], [1], [1], [0]])

brain = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
brain.train(training_data, correct_answers, train_rounds=5000)

brain.show_learning()
brain.show_decision_making(training_data, correct_answers)

print("\nTesting network:")
for example in training_data:
    prediction = brain.think(example)
    print(f"Input {example} → Prediction: {prediction[0]:.4f}")
