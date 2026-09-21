from rag import retrieve_chunks
from risk_flagging import flag_risk_statements

question = "What safety incidents have occurred at mining sites in Plateau State?"
chunks = retrieve_chunks(question, user_id=0, top_k=5)

print(f"Question: {question}\n")
print(f"Retrieved {len(chunks)} chunks.\n")

risks = flag_risk_statements(question, chunks, answer=None)

if not risks:
    print("No risk-relevant statements identified.")
else:
    print("Risk-relevant statements found:")
    for r in risks:
        print(f"  - {r}")