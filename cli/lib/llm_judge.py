from .query_enhancement import ask_llm
from .reranking import parse_json_int_list

JUDGE_DESCRIPTION_LENGTH = 300
MAX_RELEVANCE_SCORE = 3


def judge_results(query: str, results: list[dict]) -> list[int]:
    formatted_results = [
        f"{i}. {r.get('title', '')} - {r.get('document', '')[:JUDGE_DESCRIPTION_LENGTH]}"
        for i, r in enumerate(results, start=1)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    scores = parse_json_int_list(ask_llm(prompt))
    # Clamp stray values and pad if the model returned too few scores
    scores = [min(max(score, 0), MAX_RELEVANCE_SCORE) for score in scores[: len(results)]]
    return scores + [0] * (len(results) - len(scores))
