import argparse
import sys

from constants import BM25_B, BM25_K1
from lib.keyword_search import InvertedIndex
from search_utils import tokenize, tokenize_term

MAX_RESULTS = 5


def load_index() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index not found. Run the 'build' command first.")
        sys.exit(1)
    return index


def search_movies(index: InvertedIndex, query: str) -> list[dict]:
    results = []
    seen: set[int] = set()
    for token in tokenize(query):
        for doc_id in index.get_documents(token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(index.docmap[doc_id])
            if len(results) >= MAX_RESULTS:
                return results
    return results


def search_command(query: str) -> None:
    index = load_index()
    print(f"Searching for: {query}")
    results = search_movies(index, query)
    for i, movie in enumerate(results, start=1):
        print(f"{i}. {movie['title']} (ID: {movie['id']})")


def tf_command(doc_id: int, term: str) -> None:
    index = load_index()
    token = tokenize_term(term)
    print(index.get_tf(doc_id, token))


def idf_command(term: str) -> None:
    index = load_index()
    token = tokenize_term(term)
    idf = index.get_idf(token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str) -> None:
    index = load_index()
    token = tokenize_term(term)
    tf_idf = index.get_tf_idf(doc_id, token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term: str) -> float:
    index = load_index()
    token = tokenize_term(term)
    return index.get_bm25_idf(token)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    index = load_index()
    token = tokenize_term(term)
    return index.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(query: str, limit: int) -> None:
    index = load_index()
    results = index.bm25_search(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"{i}. ({result['id']}) {result['title']} - Score: {result['score']:.2f}")


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to score")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score of a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to score")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=MAX_RESULTS, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm25_search_command(args.query, args.limit)
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
