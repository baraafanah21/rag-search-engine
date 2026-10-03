import os

import numpy as np
from sentence_transformers import SentenceTransformer

from search_utils import load_movies

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = "cache"
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, dict] = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")
        return self.model.encode([text])[0]

    def __set_documents(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

    def build_embeddings(self, documents: list[dict]):
        self.__set_documents(documents)
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.__set_documents(documents)
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scored = [
            (cosine_similarity(query_embedding, doc_embedding), doc)
            for doc_embedding, doc in zip(self.embeddings, self.documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scored[:limit]
        ]


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be at least 0 and smaller than the chunk size")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i : i + chunk_size]
        # Stop once a chunk would only repeat words already in the previous chunk
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
