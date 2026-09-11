import numpy as np
import re


def parse_pg_vector(pg_vector_str):
    """pgvector returns embeddings as a string like '[0.1,0.2,...]' - convert to a list of floats."""
    return [float(x) for x in pg_vector_str.strip("[]").split(",")]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def split_sentences(text):
    """Naive sentence splitter - good enough for MVP, not perfect on abbreviations/decimals."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]

def extract_claimed_citations(sentence):
    """
    Finds mentions like 'Source 3, page 20' or '(Source 2, page 1)' and returns
    a list of (source_number, page_number) tuples. page_number may be None if not stated.
    """
    matches = re.findall(r"Source\s+(\d+)(?:,?\s*page\s+(\d+))?", sentence, re.IGNORECASE)
    results = []
    for source_str, page_str in matches:
        source_num = int(source_str)
        page_num = int(page_str) if page_str else None
        results.append((source_num, page_num))
    return results

def check_grounding(answer, chunks, embedding_model, threshold=0.55):
    sentences = split_sentences(answer)
    sentence_results = []

    for sentence in sentences:
        sentence_embedding = embedding_model.encode(sentence)
        best_chunk, best_score, best_source_number = None, -1, None

        for i, chunk in enumerate(chunks, start=1):
            chunk_vector = parse_pg_vector(chunk.embedding_vector)
            score = cosine_similarity(sentence_embedding, chunk_vector)
            if score > best_score:
                best_score, best_chunk, best_source_number = score, chunk, i

        claimed_citations = extract_claimed_citations(sentence)
        claimed_source_numbers = [c[0] for c in claimed_citations]
        claimed_page_numbers = [c[1] for c in claimed_citations if c[1] is not None]

        source_number_mismatch = (
            len(claimed_source_numbers) > 0
            and best_source_number is not None
            and best_source_number not in claimed_source_numbers
        )

        actual_page = best_chunk.page_number if best_chunk else None
        page_number_mismatch = (
            len(claimed_page_numbers) > 0
            and actual_page is not None
            and actual_page not in claimed_page_numbers
        )

        citation_mismatch = source_number_mismatch or page_number_mismatch
        sentence_results.append({
            "sentence": sentence,
            "matched_title": best_chunk.title if best_chunk else None,
            "matched_page": best_chunk.page_number if best_chunk else None,
            "actual_source_number": best_source_number,
            "claimed_source_numbers": claimed_source_numbers,
            "claimed_page_numbers": claimed_page_numbers,
            "similarity": round(best_score, 4),
            "grounded": best_score >= threshold,
            "source_number_mismatch": source_number_mismatch,
            "page_number_mismatch": page_number_mismatch,
            "citation_mismatch": citation_mismatch,
        })

    grounded_count = sum(1 for r in sentence_results if r["grounded"])
    mismatch_count = sum(1 for r in sentence_results if r["citation_mismatch"])
    total = len(sentence_results)

    if total == 0:
        status = "insufficient_evidence"
    elif grounded_count == total and mismatch_count == 0:
        status = "supported"
    elif grounded_count == 0:
        status = "unverified"
    elif mismatch_count > 0:
        status = "supported_with_citation_errors"
    else:
        status = "partially_supported"

    return status, sentence_results

