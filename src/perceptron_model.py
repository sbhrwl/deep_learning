import numpy as np


class Perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4
        print(f"self.weights: {self.weights}")
        self.eta = eta
        self.epochs = epochs

    def thatsWhatPerceptronDoes(self, inputs, weights):
        # Step 1: Dot Product
        z = np.dot(inputs, weights)
        print("Dot Product", z)
        # Step 2: Activation function
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        self.X = X
        self.y = y
        print(f"X: \n{X}")
        print(f"y: \n{y}")

        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]  # concatenation
        print(f"X_with_bias: \n{X_with_bias}")

        for epoch in range(self.epochs):
            print(f"for epoch: {epoch}")
            y_hat = self.thatsWhatPerceptronDoes(X_with_bias, self.weights)
            print(f"predicted value: \n{y_hat}")
            error = self.y - y_hat
            print(f"error: \n{error}")
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)
            print(f"updated weights: \n{self.weights}")
            print("#############\n")
