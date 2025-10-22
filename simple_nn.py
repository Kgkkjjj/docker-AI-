import random
import math

class SimpleNeuralNetwork:
    def __init__(self, n_inputs, n_hidden):
        # weights from input layer to hidden layer
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(n_inputs)] for _ in range(n_hidden)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(n_hidden)]
        # weights from hidden layer to single output neuron
        self.w2 = [random.uniform(-0.5, 0.5) for _ in range(n_hidden)]
        self.b2 = random.uniform(-0.5, 0.5)

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def predict_proba(self, x):
        # forward pass
        hidden = []
        for weights, bias in zip(self.w1, self.b1):
            z = sum(w * x_i for w, x_i in zip(weights, x)) + bias
            hidden.append(self._sigmoid(z))
        z2 = sum(w * h for w, h in zip(self.w2, hidden)) + self.b2
        return self._sigmoid(z2)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def train(self, X, y, epochs=10000, lr=0.1):
        for _ in range(epochs):
            for features, label in zip(X, y):
                # forward
                hidden = []
                hidden_z = []
                for weights, bias in zip(self.w1, self.b1):
                    z = sum(w * x_i for w, x_i in zip(weights, features)) + bias
                    hidden_z.append(z)
                    hidden.append(self._sigmoid(z))
                output_z = sum(w*h for w, h in zip(self.w2, hidden)) + self.b2
                output = self._sigmoid(output_z)

                # output error (cross-entropy derivative)
                delta_out = output - label

                # hidden layer errors
                deltas_hidden = []
                for j in range(len(self.w1)):
                    delta = self.w2[j] * delta_out * self._sigmoid_deriv(hidden[j])
                    deltas_hidden.append(delta)

                # update weights hidden->output
                for j in range(len(self.w2)):
                    self.w2[j] -= lr * delta_out * hidden[j]
                self.b2 -= lr * delta_out

                # update weights input->hidden
                for j in range(len(self.w1)):
                    for i in range(len(self.w1[j])):
                        self.w1[j][i] -= lr * deltas_hidden[j] * features[i]
                    self.b1[j] -= lr * deltas_hidden[j]

    def evaluate(self, X, y):
        correct = 0
        for features, label in zip(X, y):
            if self.predict(features) == label:
                correct += 1
        return correct / len(X)

if __name__ == "__main__":
    # XOR dataset
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y = [0, 1, 1, 0]

    nn = SimpleNeuralNetwork(n_inputs=2, n_hidden=2)
    nn.train(X, y, epochs=20000, lr=0.1)

    acc = nn.evaluate(X, y)
    print(f"Training accuracy: {acc*100:.2f}%")
    for features in X:
        prob = nn.predict_proba(features)
        pred = nn.predict(features)
        print(f"Input: {features} => Prob: {prob:.3f}, Predicted: {pred}")
