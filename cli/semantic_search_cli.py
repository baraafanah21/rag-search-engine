import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    chunk_text,
    embed_query_text,
    embed_text,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
)
from search_utils import load_movies

DESCRIPTION_PREVIEW_LENGTH = 100


def search_command(query: str, limit: int) -> None:
    search = SemanticSearch()
    search.load_or_create_embeddings(load_movies())
    results = search.search(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:DESCRIPTION_PREVIEW_LENGTH]}...")
        print()


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunk_text(text, chunk_size, overlap), start=1):
        print(f"{i}. {chunk}")


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> None:
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(semantic_chunk_text(text, max_chunk_size, overlap), start=1):
        print(f"{i}. {chunk}")


def embed_chunks_command() -> None:
    documents = load_movies()
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int) -> None:
    documents = load_movies()
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(documents)
    results = search.search_chunks(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Build or load movie embeddings and print their shape"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Search query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of words shared between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into chunks of whole sentences"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of sentences shared between chunks"
    )

    subparsers.add_parser("embed_chunks", help="Build or load embeddings for movie chunks")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies by meaning across description chunks"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
