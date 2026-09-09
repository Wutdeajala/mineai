from hybrid_search import hybrid_search_with_options
from test_set import TEST_QUERIES


def precision_at_k(retrieved_pages, relevant_pages, k=5):
    """Of the top K retrieved pages, what fraction are actually relevant?"""
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in relevant_pages)
    return hits / k


def mean_reciprocal_rank(retrieved_pages, relevant_pages):
    """1 / (rank of the first relevant result found). 0 if none found."""
    for i, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1 / i
    return 0


def run_evaluation(use_reranker):
    precisions, rr_scores = [], []

    for item in TEST_QUERIES:
        results = hybrid_search_with_options(item["question"], top_k=5, use_reranker=use_reranker)
        retrieved_pages = [chunk.page_number for chunk, score in results]

        p = precision_at_k(retrieved_pages, item["relevant_pages"])
        rr = mean_reciprocal_rank(retrieved_pages, item["relevant_pages"])

        precisions.append(p)
        rr_scores.append(rr)

        print(f"  Q: {item['question'][:55]}...")
        print(f"     Retrieved pages: {retrieved_pages} | Relevant: {item['relevant_pages']}")
        print(f"     Precision@5: {p:.2f} | Reciprocal Rank: {rr:.2f}")

    avg_precision = sum(precisions) / len(precisions)
    avg_rr = sum(rr_scores) / len(rr_scores)
    return avg_precision, avg_rr


if __name__ == "__main__":
    print("=== BASELINE (hybrid, no reranker) ===")
    baseline_p, baseline_rr = run_evaluation(use_reranker=False)

    print("\n=== ENHANCED (hybrid + reranker) ===")
    enhanced_p, enhanced_rr = run_evaluation(use_reranker=True)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25}{'Baseline':<12}{'Enhanced'}")
    print(f"{'Avg Precision@5':<25}{baseline_p:<12.3f}{enhanced_p:.3f}")
    print(f"{'Avg Reciprocal Rank':<25}{baseline_rr:<12.3f}{enhanced_rr:.3f}")