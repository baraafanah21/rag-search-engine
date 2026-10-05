import os

from dotenv import load_dotenv
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openrouter/free"


def get_llm_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def ask_llm(prompt: str) -> str:
    response = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def spell_correct_query(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    corrected = ask_llm(prompt).strip('"')
    return corrected or query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    rewritten = ask_llm(prompt).strip('"')
    return rewritten or query


def enhance_query(query: str, method: str | None) -> str:
    match method:
        case "spell":
            return spell_correct_query(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
