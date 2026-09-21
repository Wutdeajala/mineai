from rag import retrieve_chunks
from missing_info import identify_missing_information

question = "What compensation and enforcement mechanisms exist for host communities, and how effective are they in practice?"
chunks = retrieve_chunks(question, user_id=0, top_k=5)

print(f"Question: {question}\n")
print(f"Retrieved {len(chunks)} chunks.\n")

gaps = identify_missing_information(question, chunks, answer=None)

if not gaps:
    print("No significant gaps identified.")
else:
    print("Identified gaps in evidence:")
    for gap in gaps:
        print(f"  - {gap}")