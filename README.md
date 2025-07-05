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
- `layered_models.py` – demonstrates very deep networks with 20 layers for
  text samples and 30 layers for numeric data.

Each script can be executed individually, for example:

```bash
python3 perceptron.py
```

## Example: Word Dataset Processor

The processor script demonstrates the additional utilities described above. It
can load individual files or entire directories of `.txt`, `.csv`, `.json`, `.xml`,
`.html`, and `.md` data, build a vocabulary, and print TF‑IDF vectors.

### Running

```bash
python3 word_dataset_processor.py | head -n 10
```

## Example: Layered Models

The `layered_models.py` script builds extremely deep networks to demonstrate
handling of text and numeric data. It now runs a 30-layer model on short text
samples and a 30-layer model on simple numeric input.

### Running

```bash
python3 layered_models.py | head -n 10
```

## Example: Auto System

The `ai_model.py` script aggregates the algorithms and chooses one based on a task label.
Supported labels include `classification`, `regression`, `clustering`, `nlp`,
`reinforcement`, `evolution`, `text30`, `data30`, `train_csv`, and
`train_folder`. The `nlp` task can optionally accept a file or directory
path to process.

### Running

```bash
python3 ai_model.py classification
```

To process text from a directory and compute TF‑IDF vectors:

```bash
python3 ai_model.py nlp my_texts/
```

To train logistic regression on a CSV file with the label in the last column:

```bash
python3 ai_model.py train_csv data.csv
```

To run the extremely deep text model:

```bash
python3 ai_model.py text30
```
