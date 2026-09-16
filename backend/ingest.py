from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from grounding import split_sentences
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


import pdfplumber


def table_to_readable_text(table_data):
    """Converts a table's raw rows into clean, readable 'Field: value' lines - one per row."""
    if not table_data or len(table_data) < 2:
        return []

    header = [str(h).strip() if h else "" for h in table_data[0]]
    lines = []
    for row in table_data[1:]:
        parts = []
        for h, cell in zip(header, row):
            cell_str = str(cell).strip().replace("\n", " ") if cell else ""
            if h and cell_str:
                parts.append(f"{h}: {cell_str}")
        if parts:
            lines.append(". ".join(parts) + ".")
    return lines


def extract_content_by_page(pdf_path):
    """Returns two lists: (page_number, prose_text) and (page_number, table_row_text)."""
    prose_pages = []
    table_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_number = i + 1
            tables = page.find_tables()

            # Extract and convert each detected table
            for table in tables:
                raw_data = table.extract()
                readable_lines = table_to_readable_text(raw_data)
                for line in readable_lines:
                    table_rows.append((page_number, line))

            # Extract prose with table regions excluded, so tables never get jumbled into prose
            prose_page = page
            for table in tables:
                prose_page = prose_page.outside_bbox(table.bbox)

            prose_text = prose_page.extract_text()
            if prose_text and prose_text.strip():
                cleaned = clean_text(prose_text)
                if cleaned:
                    prose_pages.append((page_number, cleaned))

    return prose_pages, table_rows


def chunk_text(text_content, chunk_size=800, overlap_sentences=1):
    """Groups whole sentences into chunks, so chunks always end at real sentence boundaries."""
    sentences = split_sentences(text_content)
    chunks = []
    current_sentences = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Start the next chunk with the last sentence(s) of this one, for continuity
            current_sentences = current_sentences[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_len += len(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
def ingest_document(pdf_path, title, source_type="user_upload", mineral_tags=None, user_id=None):
    prose_pages, table_rows = extract_content_by_page(pdf_path)
    print(f"Extracted {len(prose_pages)} prose pages and {len(table_rows)} table rows.")

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

        def insert_chunk(chunk_content, page_number, chunk_type):
            nonlocal chunk_index
            embedding = model.encode(chunk_content).tolist()
            search_vec_result = conn.execute(
                text("SELECT to_tsvector('english', :chunk_text) AS sv"),
                {"chunk_text": chunk_content},
            )
            search_vec = search_vec_result.fetchone().sv

            conn.execute(
                text("""
                    INSERT INTO chunks (document_id, chunk_text, chunk_index, page_number, embedding, search_vector, chunk_type)
                    VALUES (:document_id, :chunk_text, :chunk_index, :page_number, :embedding, :search_vector, :chunk_type)
                """),
                {
                    "document_id": document_id,
                    "chunk_text": chunk_content,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "embedding": str(embedding),
                    "search_vector": search_vec,
                    "chunk_type": chunk_type,
                },
            )
            chunk_index += 1

        # Prose: chunked as before (sentence-aware), tagged 'prose'
        for page_number, page_text in prose_pages:
            for chunk_content in chunk_text(page_text):
                insert_chunk(chunk_content, page_number, "prose")

        # Table rows: each row is already a clean, complete claim - one chunk per row, tagged 'table'
        for page_number, row_text in table_rows:
            insert_chunk(row_text, page_number, "table")

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