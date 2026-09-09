from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))


def keyword_search(query, top_k=3):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    c.id, c.chunk_text, c.page_number, d.title,
                    ts_rank(c.search_vector, plainto_tsquery('english', :query)) AS rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.search_vector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT :top_k
            """),
            {"query": query, "top_k": top_k},
        )
        return result.fetchall()


if __name__ == "__main__":
    query = "large-scale mining capital investment"
    print(f"Query: {query}\n")
    results = keyword_search(query)
    for row in results:
        print(f"[Rank: {row.rank:.4f}] {row.title} (page {row.page_number})")
        print(f"  {row.chunk_text[:150]}...")
        print()