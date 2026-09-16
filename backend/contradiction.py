
"""
Contradiction detection using NLI (Natural Language Inference).

STATUS: Experimental, NOT wired into the main /chat pipeline by default.

Built and tested across two full iterations:
1. Initial version compared whole chunks -> found to produce meaningless
   results due to character-based chunking cutting sentences mid-thought.
   Fixed by switching to sentence-aware chunking (see ingest.py chunk_text).
2. Second version compared clean sentence pairs -> found that
   cross-encoder/nli-deberta-v3-small systematically over-predicts
   "contradiction" on dense academic/legal prose, including on genuinely
   compatible adjacent sentences and bare footnote fragments.

Both chunking fixes were kept (they improve retrieval and grounding
generally), but this feature itself is not reliable enough to surface to
users yet. Revisit with either a larger/legal-domain-tuned NLI model, or
an LLM-based comparison approach, before enabling by default.
"""

from sentence_transformers import CrossEncoder
import numpy as np
from grounding import parse_pg_vector, cosine_similarity, split_sentences

print("Loading contradiction detection model...")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
print("Contradiction model loaded.")

NLI_LABELS = ["contradiction", "entailment", "neutral"]


def softmax(logits):
    exp_scores = np.exp(logits - np.max(logits))
    return exp_scores / exp_scores.sum()


def find_contradictions(chunks, topical_similarity_threshold=0.5, contradiction_confidence_threshold=0.75):
    contradictions = []
    n = len(chunks)

    for i in range(n):
        for j in range(i + 1, n):
            chunk_a, chunk_b = chunks[i], chunks[j]
            if chunk_a.id == chunk_b.id:
                continue
            if chunk_a.chunk_type == "table" or chunk_b.chunk_type == "table":
                continue            

            vec_a = parse_pg_vector(chunk_a.embedding_vector)
            vec_b = parse_pg_vector(chunk_b.embedding_vector)
            topical_sim = cosine_similarity(vec_a, vec_b)

            if topical_sim < topical_similarity_threshold:
                continue

            sentences_a = split_sentences(chunk_a.chunk_text)
            sentences_b = split_sentences(chunk_b.chunk_text)

            for sent_a in sentences_a:
                for sent_b in sentences_b:
                    if len(sent_a) < 20 or len(sent_b) < 20:
                        continue

                    raw_scores = nli_model.predict([(sent_a, sent_b)])[0]
                    probabilities = softmax(raw_scores)
                    label_idx = int(np.argmax(probabilities))
                    label = NLI_LABELS[label_idx]
                    confidence = float(probabilities[label_idx])

                    if label == "contradiction" and confidence >= contradiction_confidence_threshold:
                        contradictions.append({
                            "sentence_a": sent_a,
                            "sentence_b": sent_b,
                            "chunk_a_title": chunk_a.title,
                            "chunk_a_page": chunk_a.page_number,
                            "chunk_b_title": chunk_b.title,
                            "chunk_b_page": chunk_b.page_number,
                            "confidence": round(confidence, 4),
                        })

    return contradictions