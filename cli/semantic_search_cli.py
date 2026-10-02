import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Build or load movie embeddings and print their shape"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
