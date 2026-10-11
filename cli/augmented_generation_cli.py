import argparse

from lib.augmented_generation import (
    generate_answer,
    generate_citation_answer,
    generate_question_answer,
    generate_summary,
)
from lib.hybrid_search import HybridSearch
from search_utils import load_movies

RRF_K = 60
RAG_SEARCH_LIMIT = 5


def rag_command(query: str) -> None:
    search = HybridSearch(load_movies())
    results = search.rrf_search(query, RRF_K, RAG_SEARCH_LIMIT)
    answer = generate_answer(query, results)

    print("Search Results:")
    for result in results:
        print(f"- {result['title']}")
    print()
    print("RAG Response:")
    print(answer)


def summarize_command(query: str, limit: int) -> None:
    search = HybridSearch(load_movies())
    results = search.rrf_search(query, RRF_K, limit)
    summary = generate_summary(query, results)

    print("Search Results:")
    for result in results:
        print(f"  - {result['title']}")
    print()
    print("LLM Summary:")
    print(summary)


def citations_command(query: str, limit: int) -> None:
    search = HybridSearch(load_movies())
    results = search.rrf_search(query, RRF_K, limit)
    answer = generate_citation_answer(query, results)

    print("Search Results:")
    for result in results:
        print(f"  - {result['title']}")
    print()
    print("LLM Answer:")
    print(answer)


def question_command(question: str, limit: int) -> None:
    search = HybridSearch(load_movies())
    results = search.rrf_search(question, RRF_K, limit)
    answer = generate_question_answer(question, results)

    print("Search Results:")
    for result in results:
        print(f"  - {result['title']}")
    print()
    print("Answer:")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results with an LLM"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of search results"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with citations to search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query to answer")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of search results"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question using retrieved movie documents"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of search results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)
        case "summarize":
            summarize_command(args.query, args.limit)
        case "citations":
            citations_command(args.query, args.limit)
        case "question":
            question_command(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
