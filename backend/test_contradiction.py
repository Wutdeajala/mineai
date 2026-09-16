from rag import retrieve_chunks
from contradiction import find_contradictions

question = "Do host communities have consent rights over mining on their land?"
chunks = retrieve_chunks(question, user_id=0, top_k=8)

print(f"Retrieved {len(chunks)} chunks for: {question}\n")
for c in chunks:
    print(f"  [{c.title}, page {c.page_number}] {c.chunk_text[:80]}...")

print("\nChecking for contradictions...\n")
contradictions = find_contradictions(chunks)

if not contradictions:
    print("No contradictions found above threshold.")
else:
    for c in contradictions:
        print(f"CONTRADICTION (confidence {c['confidence']}):")
        print(f"  A [{c['chunk_a_title']}, page {c['chunk_a_page']}]: {c['sentence_a']}")
        print(f"  B [{c['chunk_b_title']}, page {c['chunk_b_page']}]: {c['sentence_b']}")
        print()