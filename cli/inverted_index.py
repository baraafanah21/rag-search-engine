import math
import os
import pickle
from collections import Counter

from constants import BM25_B, BM25_K1
from search_utils import load_movies, tokenize

CACHE_DIR = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(term, set()))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(term, set()))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_ratio = doc_length / avg_doc_length if avg_doc_length > 0 else 0.0
        length_norm = 1 - b + b * length_ratio
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> list[tuple[dict, float]]:
        query_tokens = tokenize(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in query_tokens)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in ranked[:limit]]

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def build(self) -> None:
        for movie in load_movies():
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
