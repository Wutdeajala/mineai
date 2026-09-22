from ddgs import DDGS


def search_web(query, max_results=5):
    """Returns a list of dicts: title, url, snippet."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results
def format_external_results_as_evidence(results, source_number_start):
    """
    Converts web search results into the same 'Source N: ...' format our prompt
    already uses for curated evidence, but explicitly labeled as external/unverified.
    """
    evidence_block = ""
    source_map = []
    for i, r in enumerate(results, start=source_number_start):
        evidence_block += f"[Source {i}: EXTERNAL WEB SOURCE (unverified) - {r['title']}, {r['url']}]\n{r['snippet']}\n\n"
        source_map.append({"source_number": i, "title": r["title"], "url": r["url"]})
    return evidence_block, source_map


if __name__ == "__main__":
    query = "Nigerian Minerals and Mining Act 2007 recent amendments"
    print(f"Searching: {query}\n")
    results = search_web(query)
    for r in results:
        print(f"- {r['title']}")
        print(f"  {r['url']}")
        print(f"  {r['snippet'][:150]}...")
        print()