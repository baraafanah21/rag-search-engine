import argparse
import json

from lib.hybrid_search import HybridSearch
from search_utils import load_movies

GOLDEN_DATASET_PATH = "data/golden_dataset.json"
RRF_K = 60


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)["test_cases"]


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    hits = sum(1 for title in retrieved[:k] if title in relevant_set)
    return hits / k


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    search = HybridSearch(load_movies())

    print(f"k={limit}")
    for test_case in load_golden_dataset():
        query = test_case["query"]
        relevant = test_case["relevant_docs"]
        results = search.rrf_search(query, RRF_K, limit)
        retrieved = [result["title"] for result in results]
        precision = precision_at_k(retrieved, relevant, limit)

        print()
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")


if __name__ == "__main__":
    main()
