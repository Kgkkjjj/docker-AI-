import math

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = []
        self.mean = []

    def fit(self, X):
        n_samples = len(X)
        n_features = len(X[0])
        self.mean = [sum(col)/n_samples for col in zip(*X)]
        centered = [[x_j - self.mean[j] for j, x_j in enumerate(x)] for x in X]
        cov = [[0.0]*n_features for _ in range(n_features)]
        for x in centered:
            for i in range(n_features):
                for j in range(n_features):
                    cov[i][j] += x[i]*x[j]
        for i in range(n_features):
            for j in range(n_features):
                cov[i][j] /= n_samples-1
        # naive power iteration for first component
        comp = [1.0]*n_features
        for _ in range(100):
            new = [sum(cov[i][j]*comp[j] for j in range(n_features)) for i in range(n_features)]
            norm = math.sqrt(sum(v*v for v in new))
            comp = [v/norm for v in new]
        self.components = [comp[:self.n_components]]

    def transform(self, X):
        centered = [[x_j - self.mean[j] for j, x_j in enumerate(x)] for x in X]
        return [[sum(a*b for a,b in zip(comp,row)) for comp in self.components] for row in centered]

if __name__ == "__main__":
    X = [[2,0],[0,2],[3,1],[1,3]]
    pca = PCA(n_components=1)
    pca.fit(X)
    reduced = pca.transform(X)
    print("Reduced vectors:", reduced)
