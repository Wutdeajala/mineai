from hybrid_search import hybrid_search_with_options

test_queries = [
    "large-scale mining capital investment",
    "environmental impact of mining",
    "artisanal miners daily life",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print('='*70)

    print("\n--- WITHOUT reranker (baseline) ---")
    for chunk, score in hybrid_search_with_options(query, use_reranker=False):
        print(f"[{score:.4f}] page {chunk.page_number}: {chunk.chunk_text[:90]}...")

    print("\n--- WITH reranker (enhanced) ---")
    for chunk, score in hybrid_search_with_options(query, use_reranker=True):
        print(f"[{score:.4f}] page {chunk.page_number}: {chunk.chunk_text[:90]}...")