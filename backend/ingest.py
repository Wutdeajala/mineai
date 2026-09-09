from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import re

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded.")


def clean_text(text_content):
    text_content = re.sub(r"Alex Golub\. Mining\. CEA\s*\d+.*?(?=\n|$)", "", text_content)
    text_content = re.sub(
        r"This text is licensed under a Creative Commons.*?(?=\n)", "", text_content
    )
    text_content = re.sub(r"For image use please see separate credit\(s\)\.", "", text_content)
    text_content = re.sub(r"\n{3,}", "\n\n", text_content)
    return text_content.strip()


def extract_text_by_page(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text_content = page.extract_text()
        if text_content and text_content.strip():
            cleaned = clean_text(text_content)
            if cleaned:
                pages.append((i + 1, cleaned))
    return pages


def chunk_text(text_content, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text_content):
        end = start + chunk_size
        chunks.append(text_content[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_document(pdf_path, title, source_type="user_upload", mineral_tags=None, user_id=None):
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages with text.")

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO documents (title, source_type, filename, mineral_tags, user_id)
                VALUES (:title, :source_type, :filename, :mineral_tags, :user_id)
                RETURNING id
            """),
            {
                "title": title,
                "source_type": source_type,
                "filename": os.path.basename(pdf_path),
                "mineral_tags": mineral_tags or [],
                "user_id": user_id,
            },
        )
        document_id = result.fetchone()[0]
        conn.commit()
        print(f"Created document id {document_id}")

        chunk_index = 0
        for page_number, page_text in pages:
            page_chunks = chunk_text(page_text)
            for chunk in page_chunks:
                embedding = model.encode(chunk).tolist()
                search_vec_result = conn.execute(
                    text("SELECT to_tsvector('english', :chunk_text) AS sv"),
                    {"chunk_text": chunk},
                )
                search_vec = search_vec_result.fetchone().sv

                conn.execute(
                    text("""
                        INSERT INTO chunks (document_id, chunk_text, chunk_index, page_number, embedding, search_vector)
                        VALUES (:document_id, :chunk_text, :chunk_index, :page_number, :embedding, :search_vector)
                    """),
                    {
                        "document_id": document_id,
                        "chunk_text": chunk,
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "embedding": str(embedding),
                        "search_vector": search_vec,
                    },
                )
                chunk_index += 1
        conn.commit()
        print(f"Inserted {chunk_index} chunks.")
        return document_id


if __name__ == "__main__":
    ingest_document(
        pdf_path="../data/mining.pdf",
        title="Sample Test Document",
        source_type="user_upload",
        mineral_tags=["test"],
    )