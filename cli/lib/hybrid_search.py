import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

# Fetch many more candidates than requested so the merged ranking has enough overlap
SEARCH_CANDIDATE_MULTIPLIER = 500


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        candidate_limit = limit * SEARCH_CANDIDATE_MULTIPLIER
        bm25_results = self._bm25_search(query, candidate_limit)
        semantic_results = self.semantic_search.search_chunks(query, candidate_limit)

        bm25_scores = normalize_scores([r["score"] for r in bm25_results])
        semantic_scores = normalize_scores([r["score"] for r in semantic_results])

        combined: dict[int, dict] = {}
        for result, score in zip(bm25_results, bm25_scores):
            combined.setdefault(result["id"], self._new_combined_result(result["id"]))
            combined[result["id"]]["bm25_score"] = score
        for result, score in zip(semantic_results, semantic_scores):
            combined.setdefault(result["id"], self._new_combined_result(result["id"]))
            combined[result["id"]]["semantic_score"] = score

        for result in combined.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"], result["semantic_score"], alpha
            )

        ranked = sorted(combined.values(), key=lambda r: r["hybrid_score"], reverse=True)
        return ranked[:limit]

    def _new_combined_result(self, doc_id: int) -> dict:
        doc = self.document_map[doc_id]
        return {
            "id": doc_id,
            "title": doc["title"],
            "document": doc["description"],
            "bm25_score": 0.0,
            "semantic_score": 0.0,
        }

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
