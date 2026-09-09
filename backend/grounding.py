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

        claimed_sources = extract_claimed_sources(sentence)
        citation_mismatch = (
            len(claimed_sources) > 0
            and best_source_number is not None
            and best_source_number not in claimed_sources
        )

        sentence_results.append({
            "sentence": sentence,
            "matched_title": best_chunk.title if best_chunk else None,
            "matched_page": best_chunk.page_number if best_chunk else None,
            "actual_source_number": best_source_number,
            "claimed_source_numbers": claimed_sources,
            "similarity": round(best_score, 4),
            "grounded": best_score >= threshold,
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

def extract_claimed_sources(sentence):
    """Finds mentions like 'Source 3' or 'Sources 1 and 2' and returns the numbers as a list of ints."""
    matches = re.findall(r"Source\s+(\d+)", sentence, re.IGNORECASE)
    return [int(m) for m in matches]