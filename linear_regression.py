import random

class LinearRegression:
    def __init__(self):
        self.w = random.uniform(-1, 1)
        self.b = random.uniform(-1, 1)

    def predict(self, x):
        return self.w * x + self.b

    def train(self, X, y, epochs=1000, lr=0.01):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = pred - yi
                self.w -= lr * error * xi
                self.b -= lr * error

if __name__ == "__main__":
    X = list(range(10))
    y = [2 * x + 1 for x in X]

    model = LinearRegression()
    model.train(X, y, epochs=2000, lr=0.01)

    for xi in X:
        pred = model.predict(xi)
        print(f"x={xi} => y_pred={pred:.2f}")
