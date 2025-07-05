import math

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:
    def _entropy(self, labels):
        total = len(labels)
        counts = [labels.count(0), labels.count(1)]
        ent = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                ent -= p * math.log2(p)
        return ent

    def _best_split(self, X, y):
        best_gain = 0
        best_feat = None
        for feat in range(len(X[0])):
            left_y = [y[i] for i in range(len(X)) if X[i][feat] == 0]
            right_y = [y[i] for i in range(len(X)) if X[i][feat] == 1]
            if not left_y or not right_y:
                continue
            h_before = self._entropy(y)
            h_after = (len(left_y)/len(y))*self._entropy(left_y) + (len(right_y)/len(y))*self._entropy(right_y)
            gain = h_before - h_after
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
        return best_feat

    def _build(self, X, y):
        if all(label == y[0] for label in y):
            return Node(label=y[0])
        feat = self._best_split(X, y)
        if feat is None:
            majority = 1 if y.count(1) >= y.count(0) else 0
            return Node(label=majority)
        left_X = [x for x in X if x[feat] == 0]
        left_y = [y[i] for i in range(len(X)) if X[i][feat] == 0]
        right_X = [x for x in X if x[feat] == 1]
        right_y = [y[i] for i in range(len(X)) if X[i][feat] == 1]
        left = self._build(left_X, left_y)
        right = self._build(right_X, right_y)
        return Node(feature=feat, threshold=1, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y)

    def _predict(self, node, x):
        if node.label is not None:
            return node.label
        branch = node.left if x[node.feature] == 0 else node.right
        return self._predict(branch, x)

    def predict(self, x):
        return self._predict(self.root, x)

if __name__ == "__main__":
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,0]  # XOR
    tree = DecisionTree()
    tree.fit(X, y)
    for x in X:
        pred = tree.predict(x)
        print(f"Input: {x} => Predicted: {pred}")
