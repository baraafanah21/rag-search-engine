import json
import string

from nltk.stem import PorterStemmer

MOVIES_PATH = "data/movies.json"
STOP_WORDS_PATH = "data/stopwords.txt"
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
SCORE_PRECISION = 4

stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    return text


def load_stop_words() -> set[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        words = f.read().splitlines()
    return {preprocess(word) for word in words}


STOP_WORDS = load_stop_words()


def tokenize(text: str) -> list[str]:
    tokens = preprocess(text).split()
    return [stemmer.stem(token) for token in tokens if token not in STOP_WORDS]


def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token for term {term!r}, got {tokens}")
    return tokens[0]


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def format_search_result(
    doc_id: int, title: str, document: str, score: float, metadata: dict | None = None
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
