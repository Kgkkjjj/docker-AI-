# docker-AI-

This repository demonstrates how to build AI models from scratch using only Python's standard library.

## Example: Logistic Regression

The `logistic_regression.py` script implements logistic regression without external machine learning packages. It trains on a small OR gate dataset and prints predictions after training.

### Running

```bash
python3 logistic_regression.py
```

You should see output showing the probability and predicted class for each input sample.

## Example: Simple Neural Network

The `simple_nn.py` script implements a small neural network with one hidden layer. It learns the XOR gate truth table and reports the training accuracy.

### Running

```bash
python3 simple_nn.py
```


## Example: Deep Neural Network

The `complex_nn.py` script demonstrates a deeper network with two hidden layers trained on a synthetic concentric circles dataset. This shows how a neural network can learn a non-linear decision boundary.

### Running

```bash
python3 complex_nn.py
```

The script prints the training accuracy and probabilities for a few samples.

## Additional Examples

The following scripts illustrate more advanced algorithms, all implemented from scratch:

- `perceptron.py` – trains a simple perceptron on the AND gate.
- `linear_regression.py` – fits a line to synthetic data using gradient descent.
- `kmeans.py` – demonstrates k-means clustering on a small dataset.
- `naive_bayes.py` – naive Bayes text classifier for spam vs ham.
- `knn.py` – k-nearest neighbors classifier for the OR gate.
- `decision_tree.py` – builds a tiny decision tree for the XOR problem.
- `pca.py` – runs principal component analysis to reduce dimensionality.
- `q_learning.py` – tabular Q-learning in a simple grid world.
- `genetic_algorithm.py` – evolves a string to match a target phrase.
- `rnn.py` – recurrent neural network that predicts sequence parity.
- `conv_net.py` – small 1-D convolutional network with max pooling.
- `autoencoder.py` – trains an autoencoder to reconstruct binary vectors.
- `word_dataset_processor.py` – flexible text processor with loaders for
  `.txt`, `.csv`, `.json`, `.xml`, `.html`, and `.md` files. It includes
  utilities like stopword removal, stemming, n-grams, and TF-IDF vectors.

Each script can be executed individually, for example:

```bash
python3 perceptron.py
```

## Example: Word Dataset Processor

The processor script demonstrates the additional utilities described above. It
builds a vocabulary from an in-memory dataset and prints TF‑IDF vectors.

### Running

```bash
python3 word_dataset_processor.py | head -n 10
```
