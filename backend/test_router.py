from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json
from query_router import route_query

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    result = conn.execute(text("SELECT id, title, columns FROM structured_datasets"))
    rows = result.fetchall()
    available_datasets = [{"id": r.id, "title": r.title, "columns": r.columns} for r in rows]

test_questions = [
    "What is the highest gold grade recorded in the assay data?",
    "What is a drill hole?",
    "What compensation rights do host communities have?",
    "What is the average silver grade across all samples?",
]

for q in test_questions:
    result = route_query(q, available_datasets)
    print(f"Q: {q}")
    print(f"   -> {result}")
    print()