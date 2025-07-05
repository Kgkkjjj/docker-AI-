import random
import math

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centers = []

    def _distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit(self, X, epochs=10):
        self.centers = random.sample(X, self.k)
        for _ in range(epochs):
            clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = [self._distance(x, c) for c in self.centers]
                idx = distances.index(min(distances))
                clusters[idx].append(x)
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centers[i] = [sum(dim)/len(dim) for dim in zip(*cluster)]

    def predict(self, x):
        distances = [self._distance(x, c) for c in self.centers]
        return distances.index(min(distances))

if __name__ == "__main__":
    X = [[1,2], [1,3], [8,8], [9,8]]
    model = KMeans(k=2)
    model.fit(X, epochs=5)
    for x in X:
        cluster = model.predict(x)
        print(f"Point {x} => Cluster {cluster}")
