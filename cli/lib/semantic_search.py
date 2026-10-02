from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")
        return self.model.encode([text])[0]


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
