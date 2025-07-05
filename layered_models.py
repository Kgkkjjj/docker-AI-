import random
from complex_nn import DeepNeuralNetwork
from word_dataset_processor import TextProcessor


def run_text_model():
    """Train a 30-layer network on a tiny text dataset."""
    texts = [
        "Cats are wonderful pets",
        "Dogs are loyal animals",
        "Economics affects markets",
        "Finance drives business",
    ]
    labels = [1, 1, 0, 0]
    tp = TextProcessor()
    vectors = tp.fit_transform(texts)
    input_size = len(tp.vocab)
    # input + 28 hidden + output = 30 layers total
    layer_sizes = [input_size] + [8] * 28 + [1]
    net = DeepNeuralNetwork(layer_sizes)
    net.train(vectors, labels, epochs=200, lr=0.05)
    for t, v in zip(texts, vectors):
        prob = net.predict_proba(v)
        pred = net.predict(v)
        print(f"Text: {t} => {pred} ({prob:.2f})")


def run_data_model():
    """Train a 30-layer network on simple numeric data."""
    X = [[x] for x in range(10)]
    y = [1 if x > 5 else 0 for x in range(10)]
    # input + 28 hidden + output = 30 layers total
    layer_sizes = [1] + [4] * 28 + [1]
    net = DeepNeuralNetwork(layer_sizes)
    net.train(X, y, epochs=200, lr=0.05)
    for x, label in zip(X, y):
        prob = net.predict_proba(x)
        pred = net.predict(x)
        print(f"x={x[0]} label={label} => {pred} ({prob:.2f})")


if __name__ == "__main__":
    run_text_model()
    run_data_model()
