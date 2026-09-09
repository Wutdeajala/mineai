from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Optional
from ingest import ingest_document
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import shutil
import os

UPLOAD_DIR = "../data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from rag import ask

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    user_id: int | None = None


@app.get("/")
def read_root():
    return {"status": "MineAI backend is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    with engine.connect() as conn:
        if request.conversation_id is None:
            result = conn.execute(
                text("INSERT INTO conversations (title) VALUES (:title) RETURNING id"),
                {"title": request.question[:60]},
            )
            conversation_id = result.fetchone()[0]
            conn.commit()
        else:
            conversation_id = request.conversation_id

        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:conversation_id, 'user', :content)
            """),
            {"conversation_id": conversation_id, "content": request.question},
        )
        conn.commit()

        # Now conversation-aware: passes conversation_id through
        result = ask(request.question, conversation_id=conversation_id, user_id=request.user_id)

        msg_result = conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content, evidence_status)
                VALUES (:conversation_id, 'assistant', :content, :evidence_status)
                RETURNING id
            """),
            {
                "conversation_id": conversation_id,
                "content": result["answer"],
                "evidence_status": result["evidence_status"],
            },
        )
        message_id = msg_result.fetchone()[0]
        conn.commit()

        # Populate message_citations using the chunks that were actually retrieved/used
        for rank, chunk in enumerate(result.get("chunks", []), start=1):
            conn.execute(
                text("""
                    INSERT INTO message_citations (message_id, chunk_id, relevance_rank)
                    VALUES (:message_id, :chunk_id, :relevance_rank)
                """),
                {"message_id": message_id, "chunk_id": chunk.id, "relevance_rank": rank},
            )
        conn.commit()

        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "answer": result["answer"],
            "evidence_status": result["evidence_status"],
            "sentence_grounding": result["sentence_grounding"],
        }


@app.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    mineral_tags: Optional[str] = Form(None),
    user_id: Optional[int] = Form(None),
):
    # Save the uploaded file to disk first - ingest_document expects a file path
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tags_list = [t.strip() for t in mineral_tags.split(",")] if mineral_tags else []

    document_id = ingest_document(
        pdf_path=file_path,
        title=title or file.filename,
        source_type="user_upload",
        mineral_tags=tags_list,
        user_id=user_id,
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "ingested successfully",
    }    