import sys

from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
from kmeans import KMeans
from naive_bayes import NaiveBayes
from knn import KNearestNeighbors
from decision_tree import DecisionTree
from perceptron import Perceptron
from pca import PCA
from q_learning import GridWorld, QLearningAgent
from genetic_algorithm import GA
from rnn import SimpleRNN
from conv_net import ConvNet1D
from autoencoder import Autoencoder
from simple_nn import SimpleNeuralNetwork
from complex_nn import DeepNeuralNetwork
from word_dataset_processor import TextProcessor
from layered_models import run_text_model, run_data_model

class AutoSystem:
    """Select and run algorithms based on a simple task label."""

    def __init__(self, path: str | None = None):
        self.path = path

    def _load_csv(self, path: str):
        import csv
        data = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                data.append([float(x) for x in row])
        X = [r[:-1] for r in data]
        y = [int(r[-1]) for r in data]
        return X, y

    def _load_csv_folder(self, folder: str):
        import os
        all_X, all_y = [], []
        for name in os.listdir(folder):
            if name.endswith('.csv'):
                X, y = self._load_csv(os.path.join(folder, name))
                all_X.extend(X)
                all_y.extend(y)
        return all_X, all_y

    def run(self, task: str):
        task = task.lower()
        if task == "classification":
            X = [[0, 0], [0, 1], [1, 0], [1, 1]]
            y = [0, 1, 1, 1]
            model = LogisticRegression(n_features=2)
            model.train(X, y, epochs=3000)
            for x in X:
                prob = model.predict_proba(x)
                pred = model.predict(x)
                print(f"LogReg {x} => {pred} ({prob:.2f})")
        elif task == "regression":
            X = list(range(10))
            y = [2 * x + 1 for x in X]
            model = LinearRegression()
            model.train(X, y, epochs=2000, lr=0.01)
            for x in X:
                pred = model.predict(x)
                print(f"x={x} => {pred:.2f}")
        elif task == "clustering":
            X = [[1, 2], [1, 3], [8, 8], [9, 8]]
            km = KMeans(k=2)
            km.fit(X)
            for x in X:
                cluster = km.predict(x)
                print(f"Point {x} => Cluster {cluster}")
        elif task == "nlp":
            tp = TextProcessor()
            if self.path:
                texts = tp.load_path(self.path)
            else:
                texts = [
                    "The quick brown fox jumps over the lazy dog",
                    "Never jump over the lazy dog quickly",
                ]
            vectors = tp.fit_transform(texts)
            print("Vocabulary:", tp.vocab)
            for t, v in zip(texts, vectors):
                print(t)
                print([f"{x:.2f}" for x in v])
        elif task == "reinforcement":
            env = GridWorld(n_states=5)
            agent = QLearningAgent(n_states=5)
            agent.learn(env, episodes=200)
            state = 0
            steps = 0
            while state != env.end and steps < 10:
                action = agent.choose_action(state, eps=0)
                state, _ = env.step(state, action)
                print(f"State: {state}")
                steps += 1
        elif task == "evolution":
            ga = GA(pop_size=20)
            best = ga.evolve(generations=200)
            print("Evolved:", ''.join(best.genes))
        elif task == "text30":
            run_text_model()
        elif task == "data30":
            run_data_model()
        elif task == "train_csv" and self.path:
            X, y = self._load_csv(self.path)
            model = LogisticRegression(n_features=len(X[0]))
            model.train(X, y, epochs=3000)
            for x, label in zip(X, y):
                pred = model.predict(x)
                print(f"{x} label={label} => {pred}")
        elif task == "train_folder" and self.path:
            X, y = self._load_csv_folder(self.path)
            model = LogisticRegression(n_features=len(X[0]))
            model.train(X, y, epochs=3000)
            correct = sum(model.predict(x) == y_i for x, y_i in zip(X, y))
            acc = correct / len(y) * 100
            print(f"Folder training accuracy: {acc:.1f}%")
        else:
            print(
                "Unknown task. Available tasks: classification, regression, clustering, nlp, reinforcement, evolution, text30, data30, train_csv, train_folder"
            )

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "classification"
    path = sys.argv[2] if len(sys.argv) > 2 else None
    AutoSystem(path).run(task)
