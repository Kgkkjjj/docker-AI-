import math
import csv
import json
import re
import xml.etree.ElementTree as ET
from typing import List

class TextProcessor:
    """Text processor with loaders and basic NLP utilities.

    The processor can read datasets from six file formats and provides
    several text analysis helpers including stopword removal, stemming,
    n-gram generation and TF-IDF computation.
    """

    def __init__(self):
        self.vocab: List[str] = []
        self.word_to_idx = {}
        self.idf = {}
        self.stopwords = {
            'the', 'a', 'and', 'or', 'to', 'of', 'in', 'on', 'is', 'are',
            'for', 'with', 'an', 'as'
        }

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

    # ------------------------------------------------------------------
    # Processing helpers (6 systems)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Return tokens not in the stopword set."""
        return [t for t in tokens if t not in self.stopwords]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Very small stemmer removing common suffixes."""
        stems = []
        for t in tokens:
            for suf in ('ing', 'ed', 's'):
                if t.endswith(suf) and len(t) > len(suf) + 2:
                    t = t[:-len(suf)]
                    break
            stems.append(t)
        return stems

    def bigrams(self, tokens: List[str]) -> List[str]:
        return ['{}_{}'.format(tokens[i], tokens[i+1])
                for i in range(len(tokens)-1)]

    def trigrams(self, tokens: List[str]) -> List[str]:
        return ['{}_{}_{}'.format(tokens[i], tokens[i+1], tokens[i+2])
                for i in range(len(tokens)-2)]

    def term_frequency(self, tokens: List[str]):
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Dataset loaders supporting six file types (6 systems)

    def load_txt(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as fh:
            return [line.strip() for line in fh if line.strip()]

    def load_csv(self, path: str, text_column: int = 0) -> List[str]:
        rows = []
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) > text_column:
                    rows.append(row[text_column])
        return rows

    def load_json(self, path: str, key: str = 'text') -> List[str]:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return [item[key] for item in data if key in item]

    def load_xml(self, path: str, tag: str = 'text') -> List[str]:
        tree = ET.parse(path)
        root = tree.getroot()
        return [elem.text or '' for elem in root.iter(tag)]

    def load_html(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        text = re.sub('<[^<]+?>', ' ', text)
        return [text.strip()]

    def load_markdown(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        text = re.sub(r'[#*>`]', ' ', text)
        return [text.strip()]

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
