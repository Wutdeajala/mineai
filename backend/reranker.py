from sentence_transformers import CrossEncoder

print("Loading reranker model...")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")
print("Reranker loaded.")


def rerank(query, candidates, top_k=3):
    """
    candidates: list of (chunk, score) tuples from hybrid_search, or just chunk rows.
    Returns the same chunks, reordered by the cross-encoder's relevance judgment.
    """
    chunks = [c[0] if isinstance(c, tuple) else c for c in candidates]
    pairs = [(query, chunk.chunk_text) for chunk in chunks]

    scores = reranker_model.predict(pairs)

    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]