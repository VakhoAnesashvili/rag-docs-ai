import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.db import repository as repo
from app.db.connection import get_conn, init_schema
from app.ingestion.embedder import Embedder
from app.ingestion.parsers import ParseError
from app.ingestion.service import IngestionService
from app.rag.agent import AgentPipeline
from app.rag.llm import LLMClient, LLMError
from app.rag.pipeline import RagPipeline
from app.search.hybrid import HybridSearcher

log = logging.getLogger("api")
ALLOWED = {".md", ".txt", ".pdf", ".docx", ".jsonl"}
MAX_UPLOAD_MB = 10


class State:
    searcher: HybridSearcher
    pipeline: RagPipeline
    agent: AgentPipeline
    ingest: IngestionService


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    embedder = Embedder()                       # ერთხელ იტვირთება და ყველა მოთხოვნა იყენებს
    llm = LLMClient()
    state.searcher = HybridSearcher(embedder)
    state.pipeline = RagPipeline(state.searcher, llm)
    state.agent = AgentPipeline(state.searcher, llm)
    state.ingest = IngestionService(embedder)
    yield


app = FastAPI(title="Document Intelligence RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["pipeline", "agent"] = "pipeline"


def _source(s) -> dict:
    return {"number": s.number, "chunk_id": s.chunk_id, "doc_id": s.doc_id,
            "title": s.title, "text": s.text}


@app.get("/health")
def health():
    try:
        with get_conn() as conn:
            return {"status": "ok", "counts": repo.count_rows(conn)}
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {e}")


@app.get("/documents")
def list_documents(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   q: str = ""):
    with get_conn() as conn:
        total, items = repo.list_documents(conn, limit, offset, q)
    return {"total": total, "items": items}


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"unsupported format '{suffix}', allowed: {sorted(ALLOWED)}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
        try:
            results = state.ingest.ingest_file(path)
        except ParseError as e:
            raise HTTPException(422, str(e))
    state.searcher._bm25 = None                              # BM25 თავიდან აიგება
    return {"results": [r.__dict__ for r in results]}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not state.ingest.delete(doc_id):
        raise HTTPException(404, "document not found")
    state.searcher._bm25 = None
    return {"deleted": doc_id}


@app.get("/sources/{chunk_id:path}")
def get_source(chunk_id: str):
    with get_conn() as conn:
        chunk = repo.get_chunk(conn, chunk_id)
    if not chunk:
        raise HTTPException(404, "chunk not found")
    return chunk


@app.post("/ask")
def ask(req: AskRequest):
    runner = state.agent if req.mode == "agent" else state.pipeline
    try:
        r = runner.ask(req.question)
    except LLMError as e:
        raise HTTPException(502, f"LLM unavailable: {e}")
    return {
        "question": r.question, "answer": r.answer, "abstained": r.abstained,
        "conflict": r.conflict, "conflict_note": r.conflict_note,
        "sources": [_source(s) for s in r.sources],
        "retrieved": [_source(s) for s in r.retrieved],
        "sub_queries": r.sub_queries, "trace": r.trace, "error": r.error,
    }