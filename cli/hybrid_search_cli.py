import argparse
import logging
import sys

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.query_enhancement import enhance_query
from lib.reranking import rerank
from search_utils import load_movies

DESCRIPTION_PREVIEW_LENGTH = 100
# Re-rank a wider pool than requested so the LLM can promote lower-ranked matches
RERANK_CANDIDATE_MULTIPLIER = 5

logger = logging.getLogger("hybrid_search")


def enable_debug_logging() -> None:
    # Only our logger, so library loggers (httpx, transformers) stay quiet
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[DEBUG] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def log_results(stage: str, results: list[dict]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(f"{stage} ({len(results)} results):")
    for i, result in enumerate(results, start=1):
        logger.debug(f"  {i}. ({result['id']}) {result['title']}")


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


def rrf_search_command(
    query: str, k: int, limit: int, enhance: str | None, rerank_method: str | None
) -> None:
    logger.debug(f"Original query: '{query}'")
    if enhance:
        enhanced_query = enhance_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query
    logger.debug(f"Query after enhancement ({enhance or 'none'}): '{query}'")

    search = HybridSearch(load_movies())

    if rerank_method:
        results = search.rrf_search(query, k, limit * RERANK_CANDIDATE_MULTIPLIER)
        log_results("RRF search results", results)
        print(f"Re-ranking top {limit} results using {rerank_method} method...")
        results = rerank(query, results, rerank_method)
        log_results(f"Final results after {rerank_method} re-ranking", results[:limit])
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
        for i, result in enumerate(results[:limit], start=1):
            print(f"\n{i}. {result['title']}")
            match rerank_method:
                case "batch":
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                case "cross_encoder":
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                case _:
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
            print(f"   RRF Score: {result['rrf_score']:.3f}")
            print(
                f"   BM25 Rank: {format_rank(result['bm25_rank'])}, "
                f"Semantic Rank: {format_rank(result['semantic_rank'])}"
            )
            print(f"   {result['document'][:DESCRIPTION_PREVIEW_LENGTH]}...")
        return

    results = search.rrf_search(query, k, limit)
    log_results("RRF search results", results)
    log_results("Final results (no re-ranking)", results[:limit])
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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM re-ranking method",
    )
    rrf_parser.add_argument(
        "--debug", action="store_true", help="Log each pipeline stage to stderr"
    )

    args = parser.parse_args()
    if getattr(args, "debug", False):
        enable_debug_logging()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
