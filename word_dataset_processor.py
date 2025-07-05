import math
from typing import List

class TextProcessor:
    """Simple text processor that computes TF-IDF vectors."""

    def __init__(self):
        self.vocab: List[str] = []
        self.word_to_idx = {}
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split text into alphabetic tokens."""
        tokens = []
        word = ''
        for ch in text.lower():
            if ch.isalpha():
                word += ch
            else:
                if word:
                    tokens.append(word)
                    word = ''
        if word:
            tokens.append(word)
        return tokens

    def fit(self, texts: List[str]):
        doc_counts = {}
        for text in texts:
            tokens = self._tokenize(text)
            unique = set(tokens)
            for token in unique:
                doc_counts[token] = doc_counts.get(token, 0) + 1
        self.vocab = sorted(doc_counts.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n_docs = len(texts)
        self.idf = {w: math.log(n_docs / (1 + doc_counts[w])) for w in self.vocab}

    def transform(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            total = len(tokens)
            counts = {}
            for tok in tokens:
                if tok in self.word_to_idx:
                    counts[tok] = counts.get(tok, 0) + 1
            vec = [0.0 for _ in self.vocab]
            for tok, count in counts.items():
                idx = self.word_to_idx[tok]
                tf = count / total if total else 0
                vec[idx] = tf * self.idf[tok]
            vectors.append(vec)
        return vectors

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        self.fit(texts)
        return self.transform(texts)

if __name__ == '__main__':
    dataset = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "Foxes are quick and dogs are lazy",
        "Dog owners love quick pets",
    ]
    processor = TextProcessor()
    vectors = processor.fit_transform(dataset)
    print("Vocabulary:", processor.vocab)
    print("IDF:", [f"{w}:{processor.idf[w]:.2f}" for w in processor.vocab])
    for text, vec in zip(dataset, vectors):
        print(text)
        print([f"{v:.2f}" for v in vec])
