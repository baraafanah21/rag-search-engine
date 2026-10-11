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


def generate_citation_answer(query: str, results: list[dict]) -> str:
    documents = "\n\n".join(
        f"[{index}] {result['title']}: {result['document']}"
        for index, result in enumerate(results, start=1)
    )
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return ask_llm(prompt)


def generate_question_answer(question: str, results: list[dict]) -> str:
    context = format_documents(results)
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    return ask_llm(prompt)
