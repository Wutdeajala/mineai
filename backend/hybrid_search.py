from retrieve import retrieve as embedding_search  # reuse Phase 3's function, renamed on import
from keyword_search import keyword_search


def reciprocal_rank_fusion(embedding_results, keyword_results, k=60, top_k=5):
    scores = {}
    chunk_lookup = {}

    for rank, row in enumerate(embedding_results, start=1):
        scores[row.id] = scores.get(row.id, 0) + 1 / (k + rank)
        chunk_lookup[row.id] = row

    for rank, row in enumerate(keyword_results, start=1):
        scores[row.id] = scores.get(row.id, 0) + 1 / (k + rank)
        chunk_lookup[row.id] = row

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [(chunk_lookup[cid], scores[cid]) for cid in ranked_ids[:top_k]]


def hybrid_search(query, top_k=5):
    embedding_results = embedding_search(query, top_k=10)
    keyword_results = keyword_search(query, top_k=10)
    return reciprocal_rank_fusion(embedding_results, keyword_results, top_k=top_k)


def hybrid_search_with_options(query, top_k=3, use_reranker=False):
    candidates = hybrid_search(query, top_k=10)  # wider pool for reranker to work with

    if use_reranker:
        from reranker import rerank
        return rerank(query, candidates, top_k=top_k)
    else:
        return candidates[:top_k]        


if __name__ == "__main__":
    query = "large-scale mining capital investment"

    print("=== WITHOUT reranker (baseline) ===")
    for chunk, score in hybrid_search_with_options(query, use_reranker=False):
        print(f"[{score:.4f}] page {chunk.page_number}: {chunk.chunk_text[:100]}...")

    print("\n=== WITH reranker (enhanced) ===")
    for chunk, score in hybrid_search_with_options(query, use_reranker=True):
        print(f"[{score:.4f}] page {chunk.page_number}: {chunk.chunk_text[:100]}...")    