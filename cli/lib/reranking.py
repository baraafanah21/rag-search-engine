import json
import re
import time

from sentence_transformers import CrossEncoder

from .query_enhancement import ask_llm

# Pause between LLM calls to stay under OpenRouter's free-tier rate limit
LLM_CALL_DELAY_SECONDS = 3
BATCH_DESCRIPTION_LENGTH = 300
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


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


def parse_ranked_ids(text: str) -> list[int]:
    # Some models wrap JSON in a Markdown code block despite being told not to
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return [int(doc_id) for doc_id in json.loads(text.strip())]


def rerank_batch(query: str, docs: list[dict]) -> list[dict]:
    doc_list_str = "\n".join(
        f"{doc['id']}: {doc.get('title', '')} - {doc.get('document', '')[:BATCH_DESCRIPTION_LENGTH]}"
        for doc in docs
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    ranked_ids = parse_ranked_ids(ask_llm(prompt))

    rank_by_id: dict[int, int] = {}
    for doc_id in ranked_ids:
        rank_by_id.setdefault(doc_id, len(rank_by_id) + 1)
    # Movies the LLM left out keep their RRF order after the ranked ones
    for doc in docs:
        rank_by_id.setdefault(doc["id"], len(rank_by_id) + 1)

    for doc in docs:
        doc["rerank_rank"] = rank_by_id[doc["id"]]
    return sorted(docs, key=lambda d: d["rerank_rank"])


def rerank_cross_encoder(query: str, docs: list[dict]) -> list[dict]:
    pairs = [[query, f"{doc.get('title', '')} - {doc.get('document', '')}"] for doc in docs]
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    scores = cross_encoder.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["cross_encoder_score"] = float(score)
    return sorted(docs, key=lambda d: d["cross_encoder_score"], reverse=True)


def rerank(query: str, docs: list[dict], method: str) -> list[dict]:
    match method:
        case "individual":
            return rerank_individual(query, docs)
        case "batch":
            return rerank_batch(query, docs)
        case "cross_encoder":
            return rerank_cross_encoder(query, docs)
        case _:
            raise ValueError(f"Unknown rerank method: {method}")
