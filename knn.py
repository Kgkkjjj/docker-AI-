import math

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X = []
        self.y = []

    def _distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        distances = [self._distance(x, xi) for xi in self.X]
        pairs = sorted(zip(distances, self.y))[:self.k]
        votes = sum(label for _, label in pairs)
        return 1 if votes >= self.k / 2 else 0

if __name__ == "__main__":
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,1]  # OR gate

    knn = KNearestNeighbors(k=3)
    knn.train(X, y)
    for x in X:
        pred = knn.predict(x)
        print(f"Input: {x} => Predicted: {pred}")
