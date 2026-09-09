import ollama
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

from grounding import check_grounding

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def retrieve_chunks(query, user_id=None, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    c.id, c.chunk_text, c.page_number, d.title,
                    c.embedding AS embedding_vector,
                    c.embedding <=> :query_embedding AS distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.source_type = 'curated_kb'
                   OR (d.source_type = 'user_upload' AND d.user_id = :user_id)
                ORDER BY distance
                LIMIT :top_k
            """),
            {"query_embedding": str(query_embedding), "top_k": top_k, "user_id": user_id},
        )
        return result.fetchall()


def get_conversation_history(conversation_id, engine, limit=6):
    """Fetch the most recent messages in a conversation, oldest first."""
    if conversation_id is None:
        return []

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = :conversation_id
                ORDER BY id DESC
                LIMIT :limit
            """),
            {"conversation_id": conversation_id, "limit": limit},
        )
        rows = result.fetchall()
        return list(reversed(rows))


def build_prompt(question, chunks, history=None):
    evidence_block = ""
    for i, chunk in enumerate(chunks, start=1):
        evidence_block += f"[Source {i}: {chunk.title}, page {chunk.page_number}]\n{chunk.chunk_text}\n\n"

    history_block = ""
    if history:
        for msg in history:
            speaker = "User" if msg.role == "user" else "MineAI"
            history_block += f"{speaker}: {msg.content}\n"

    conversation_section = f"Conversation so far:\n{history_block}\n" if history_block else ""

    prompt = f"""You are MineAI, a mining industry assistant. Answer the question using ONLY the evidence provided below. Do not use any outside knowledge.

Rules:
- If the evidence fully answers the question, answer it clearly and cite which source number(s) you used.
- If the evidence only partially answers the question, say what is supported and explicitly state what is missing.
- If the evidence does not contain relevant information to answer the question, say so directly instead of guessing.
- Never invent a source or a fact not present in the evidence below.
- Use the conversation history only to understand context (e.g. what "it" or "that" refers to) — do not treat prior answers as evidence themselves.

{conversation_section}Evidence:
{evidence_block}

Question: {question}

Answer:"""
    return prompt


def ask(question, conversation_id=None, user_id=None, top_k=3):
    chunks = retrieve_chunks(question, user_id=user_id, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant documents found to answer this question.",
            "evidence_status": "insufficient_evidence",
            "sentence_grounding": [],
        }

    history = get_conversation_history(conversation_id, engine)
    prompt = build_prompt(question, chunks, history=history)

  
    response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}], options={"num_predict": 300},)
    answer = response["message"]["content"]

    status, sentence_grounding = check_grounding(answer, chunks, embedding_model)

    return {
        "answer": answer,
        "evidence_status": status,
        "sentence_grounding": sentence_grounding,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = ask("What is the difference between large-scale and small-scale mining?", conversation_id=1)
    print("ANSWER:\n", result["answer"])
    print(f"\nEVIDENCE STATUS: {result['evidence_status']}")