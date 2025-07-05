from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, X, y):
        for text, label in zip(X, y):
            self.class_counts[label] += 1
            for word in text.split():
                self.vocab.add(word)
                self.feature_counts[label][word] += 1

    def predict(self, text):
        scores = {}
        total_docs = sum(self.class_counts.values())
        for label in self.class_counts:
            prior = self.class_counts[label] / total_docs
            score = prior
            for word in text.split():
                word_freq = self.feature_counts[label][word] + 1
                denom = sum(self.feature_counts[label].values()) + len(self.vocab)
                score *= word_freq / denom
            scores[label] = score
        return max(scores, key=scores.get)

if __name__ == "__main__":
    X = ["spam offer secret", "secret click secret", "sport play win", "play sports today"]
    y = ["spam", "spam", "ham", "ham"]

    nb = NaiveBayes()
    nb.train(X, y)
    test = "secret offer today"
    pred = nb.predict(test)
    print(f"Text: '{test}' => Predicted class: {pred}")
