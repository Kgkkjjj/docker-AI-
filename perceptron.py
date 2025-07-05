import random

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.5, 0.5)

    def predict(self, x):
        activation = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, X, y, epochs=20, lr=0.1):
        for _ in range(epochs):
            for features, label in zip(X, y):
                pred = self.predict(features)
                error = label - pred
                for i in range(len(self.weights)):
                    self.weights[i] += lr * error * features[i]
                self.bias += lr * error

if __name__ == "__main__":
    # AND gate
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y = [0, 0, 0, 1]

    p = Perceptron(2)
    p.train(X, y, epochs=10, lr=0.1)

    for features in X:
        pred = p.predict(features)
        print(f"Input: {features} => Predicted: {pred}")
