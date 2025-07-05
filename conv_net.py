import random
import math

class ConvNet1D:
    def __init__(self, filter_size):
        self.filter = [random.uniform(-0.5,0.5) for _ in range(filter_size)]
        self.bias = random.uniform(-0.5,0.5)
        self.fc_w = random.uniform(-0.5,0.5)
        self.fc_b = random.uniform(-0.5,0.5)

    def _convolve(self, x):
        out = []
        for i in range(len(x)-len(self.filter)+1):
            val = sum(f*x[i+j] for j,f in enumerate(self.filter)) + self.bias
            out.append(max(0,val))  # ReLU
        return max(out)  # global max pooling

    def predict_proba(self, x):
        feat = self._convolve(x)
        z = feat * self.fc_w + self.fc_b
        return 1/(1+math.exp(-z))

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def train(self, X, y, epochs=200, lr=0.01):
        for _ in range(epochs):
            for x_i, y_i in zip(X, y):
                feat = self._convolve(x_i)
                prob = self.predict_proba(x_i)
                error = prob - y_i
                # gradients
                self.fc_w -= lr * error * feat
                self.fc_b -= lr * error
                # backprop through conv filter (approx via single max position)
                idx = max(range(len(x_i)-len(self.filter)+1), key=lambda i: sum(self.filter[j]*x_i[i+j] for j in range(len(self.filter))))
                for j in range(len(self.filter)):
                    grad = error * self.fc_w * x_i[idx+j]
                    self.filter[j] -= lr * grad
                self.bias -= lr * error

if __name__ == "__main__":
    # sequences with high sum vs low sum
    X = [[0,0,0,1],[1,1,1,1],[0,1,0,1],[1,0,1,0]]
    y = [0,1,0,1]
    net = ConvNet1D(filter_size=2)
    net.train(X, y, epochs=200)
    for seq in X:
        pred = net.predict(seq)
        print(f"Seq: {seq} => Predicted: {pred}")
