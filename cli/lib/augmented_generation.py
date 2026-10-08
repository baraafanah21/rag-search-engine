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
