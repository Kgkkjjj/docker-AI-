import random
import math

class LogisticRegression:
    def __init__(self, n_features):
        # Initialize weights and bias randomly
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = random.uniform(-0.5, 0.5)

    def _sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, x):
        # Calculate probability for a single sample
        z = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def predict(self, x):
        # Convert probability to class label
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def train(self, X, y, epochs=1000, lr=0.1):
        # Basic gradient descent training loop
        for _ in range(epochs):
            for features, label in zip(X, y):
                prediction = self.predict_proba(features)
                error = prediction - label
                # Update weights and bias
                for j in range(len(self.weights)):
                    self.weights[j] -= lr * error * features[j]
                self.bias -= lr * error

if __name__ == "__main__":
    # Simple OR gate dataset
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y = [0, 1, 1, 1]

    model = LogisticRegression(n_features=2)
    model.train(X, y, epochs=5000, lr=0.1)

    for features in X:
        prob = model.predict_proba(features)
        pred = model.predict(features)
        print(f"Input: {features} => Prob: {prob:.3f}, Predicted: {pred}")

