import re
import time

from .query_enhancement import ask_llm

# Pause between LLM calls to stay under OpenRouter's free-tier rate limit
LLM_CALL_DELAY_SECONDS = 3


def parse_score(text: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return min(max(float(match.group()), 0.0), 10.0)


def score_document(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    return parse_score(ask_llm(prompt))


def rerank_individual(query: str, docs: list[dict]) -> list[dict]:
    for i, doc in enumerate(docs):
        if i > 0:
            time.sleep(LLM_CALL_DELAY_SECONDS)
        doc["rerank_score"] = score_document(query, doc)
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)


def rerank(query: str, docs: list[dict], method: str) -> list[dict]:
    match method:
        case "individual":
            return rerank_individual(query, docs)
        case _:
            raise ValueError(f"Unknown rerank method: {method}")
