import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from search_utils import load_movies

DESCRIPTION_PREVIEW_LENGTH = 100


def normalize_command(scores: list[float]) -> None:
    for score in normalize_scores(scores):
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    search = HybridSearch(load_movies())
    results = search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results[:limit], start=1):
        print(f"{i}. {result['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
        print(f"  {result['document'][:DESCRIPTION_PREVIEW_LENGTH]}...")


def format_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def rrf_search_command(query: str, k: int, limit: int) -> None:
    search = HybridSearch(load_movies())
    results = search.rrf_search(query, k, limit)
    for i, result in enumerate(results[:limit], start=1):
        print(f"{i}. {result['title']}")
        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(
            f"  BM25 Rank: {format_rank(result['bm25_rank'])}, "
            f"Semantic Rank: {format_rank(result['semantic_rank'])}"
        )
        print(f"  {result['document'][:DESCRIPTION_PREVIEW_LENGTH]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Search movies by combining BM25 and semantic scores"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Keyword weight (1.0 = all BM25, 0.0 = all semantic)"
    )
    weighted_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Search movies by combining BM25 and semantic ranks with RRF"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k constant")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
