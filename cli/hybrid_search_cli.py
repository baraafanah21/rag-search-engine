import argparse

from lib.hybrid_search import normalize_scores


def normalize_command(scores: list[float]) -> None:
    for score in normalize_scores(scores):
        print(f"* {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
