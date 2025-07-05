import random
import math

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            w = [[random.uniform(-0.5, 0.5) for _ in range(in_size)] for _ in range(out_size)]
            b = [random.uniform(-0.5, 0.5) for _ in range(out_size)]
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):
        return max(0.0, z)

    def _relu_deriv(self, a):
        return 1.0 if a > 0 else 0.0

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def _forward(self, x):
        activations = [x]
        a = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = []
            for weights, bias in zip(w, b):
                z_val = sum(w_i * a_i for w_i, a_i in zip(weights, a)) + bias
                z.append(z_val)
            if idx == len(self.weights) - 1:
                a = [self._sigmoid(v) for v in z]
            else:
                a = [self._relu(v) for v in z]
            activations.append(a)
        return activations

    def predict_proba(self, x):
        activations = self._forward(x)
        return activations[-1][0]

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def _backward(self, activations, y):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        # output layer delta (assuming binary classification)
        delta = [activations[-1][0] - y]
        grads_w[-1] = [[delta[0] * a for a in activations[-2]]]
        grads_b[-1] = delta[:]
        # backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            next_w = self.weights[l+1]
            new_delta = []
            for j in range(len(self.weights[l])):
                error = sum(next_w[k][j] * delta[k] for k in range(len(delta)))
                new_delta.append(error * self._relu_deriv(activations[l+1][j]))
            delta = new_delta
            grads_w[l] = [[delta[i] * activations[l][j] for j in range(len(activations[l]))] for i in range(len(delta))]
            grads_b[l] = delta[:]
        return grads_w, grads_b

    def train(self, X, y, epochs=1000, lr=0.1):
        for _ in range(epochs):
            for x_i, y_i in zip(X, y):
                activations = self._forward(x_i)
                grads_w, grads_b = self._backward(activations, y_i)
                for l in range(len(self.weights)):
                    for i in range(len(self.weights[l])):
                        for j in range(len(self.weights[l][i])):
                            self.weights[l][i][j] -= lr * grads_w[l][i][j]
                        self.biases[l][i] -= lr * grads_b[l][i]

    def evaluate(self, X, y):
        correct = 0
        for x_i, y_i in zip(X, y):
            if self.predict(x_i) == y_i:
                correct += 1
        return correct / len(X)


def make_circles(n_samples=200, noise=0.1):
    X = []
    y = []
    for _ in range(n_samples // 2):
        theta = random.uniform(0, 2 * math.pi)
        r = 1 + random.uniform(-noise, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(0)
    for _ in range(n_samples // 2):
        theta = random.uniform(0, 2 * math.pi)
        r = 2 + random.uniform(-noise, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(1)
    return X, y

if __name__ == "__main__":
    X, y = make_circles(n_samples=200, noise=0.1)
    nn = DeepNeuralNetwork([2, 4, 4, 1])
    nn.train(X, y, epochs=1000, lr=0.05)
    acc = nn.evaluate(X, y)
    print(f"Training accuracy: {acc * 100:.2f}%")
    for i in range(5):
        prob = nn.predict_proba(X[i])
        pred = nn.predict(X[i])
        print(f"Sample {i}: Prob={prob:.3f}, Pred={pred}")

