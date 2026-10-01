import argparse
import sys

from inverted_index import InvertedIndex
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

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
