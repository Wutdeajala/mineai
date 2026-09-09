from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded.")


def retrieve(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    c.id,
                    c.chunk_text,
                    c.page_number,
                    d.title,
                    c.embedding <=> :query_embedding AS distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY distance
                LIMIT :top_k
            """),
            {"query_embedding": str(query_embedding), "top_k": top_k},
        )
        return result.fetchall()


if __name__ == "__main__":
    question = "What is the difference between large-scale and small-scale mining?"
    print(f"\nQuestion: {question}\n")

    results = retrieve(question)
    for row in results:
        print(f"[Distance: {row.distance:.4f}] {row.title} (page {row.page_number})")
        print(f"  {row.chunk_text[:150]}...")
        print()