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

class AutoSystem:
    """Select and run algorithms based on a simple task label."""

    def __init__(self):
        pass

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
            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Never jump over the lazy dog quickly",
            ]
            tp = TextProcessor()
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
        else:
            print("Unknown task. Available tasks: classification, regression, clustering, nlp, reinforcement, evolution")

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "classification"
    AutoSystem().run(task)
