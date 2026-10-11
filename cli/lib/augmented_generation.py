from .query_enhancement import ask_llm


def format_documents(results: list[dict]) -> str:
    return "\n\n".join(f"- {r['title']}: {r['document']}" for r in results)


def generate_answer(query: str, results: list[dict]) -> str:
    docs = format_documents(results)
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
    return ask_llm(prompt)


def generate_summary(query: str, results: list[dict]) -> str:
    docs = format_documents(results)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return ask_llm(prompt)
