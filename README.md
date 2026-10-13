Retrieval-Augmented Generation (RAG)

A Python-based movie search engine that combines keyword search, semantic search, and LLM generation to deliver relevant results and context-aware answers.

Features

* Keyword Search: BM25-based document retrieval.
* Semantic Search: Embedding-based similarity search.
* Hybrid Search: Combines keyword and semantic rankings using Reciprocal Rank Fusion (RRF).
* Query Enhancement: Spell correction, query rewriting, and query expansion.
* Re-ranking: Improves result relevance using advanced ranking techniques.
* Evaluation: Measures precision, recall, and F1 score.
* Chunking: Supports semantic chunking and chunk-level retrieval.
* RAG: Generates natural-language answers using retrieved documents as context.

Tech Stack

* Python
* NumPy
* Sentence Transformers
* OpenAI SDK
* OpenRouter API
* uv

Getting Started

Install dependencies:

uv sync

Configure your OpenRouter API key in a .env file:

OPENROUTER_API_KEY=your_api_key_here

Usage

Search for movies using BM25:

uv run cli/keyword_search_cli.py bm25search "dinosaur"

Perform semantic search:

uv run cli/semantic_search_cli.py search_chunked "dinosaur movies"

Run hybrid search with RRF:

uv run cli/hybrid_search_cli.py rrf-search "dinosaur movies"

Generate an answer using RAG:

uv run cli/augmented_generation_cli.py rag "dinosaur movies"

How It Works

1. Process the user’s query.
2. Retrieve relevant documents through keyword and semantic search.
3. Combine rankings using RRF.
4. Re-rank candidates to improve relevance.
5. Evaluate retrieval quality.
6. Provide retrieved documents to an LLM to generate a context-aware response.

Key Concepts

* Information Retrieval
* TF-IDF and BM25
* Embeddings and Cosine Similarity
* Vector Search and Chunking
* Hybrid Search and Reciprocal Rank Fusion
* Re-ranking and Search Evaluation
* Retrieval-Augmented Generation