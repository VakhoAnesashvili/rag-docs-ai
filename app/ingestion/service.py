"""დოკუმენტის დამატება / განახლება / წაშლა ერთ ადგილას."""
from dataclasses import dataclass
from pathlib import Path

from app.db import repository as repo
from app.db.connection import get_conn
from app.ingestion.chunking import StructuralChunker
from app.ingestion.embedder import Embedder
from app.ingestion.parsers import parse_file


@dataclass
class IngestResult:
    doc_id: str
    title: str
    status: str          # created / updated / unchanged
    chunks: int


class IngestionService:
    def __init__(self, embedder: Embedder, strategy: str = "structural") -> None:
        self.embedder = embedder
        self.strategy = strategy
        self.chunker = StructuralChunker()

    def ingest_file(self, path: Path) -> list[IngestResult]:
        results = []
        with get_conn() as conn:
            for doc in parse_file(path):
                old_hash = repo.get_document_hash(conn, doc.doc_id)
                if old_hash == doc.content_hash:
                    results.append(IngestResult(doc.doc_id, doc.title, "unchanged", 0))
                    continue
                status = "created" if old_hash is None else "updated"
                if old_hash is not None:
                    repo.delete_document(conn, doc.doc_id)      # ძველი chunk-ები cascade-ით იშლება
                repo.upsert_document(conn, doc)
                chunks = self.chunker.chunk(doc)
                vectors = self.embedder.embed_documents(
                    [f"{c.metadata['title']}. {c.text}" for c in chunks])
                repo.insert_chunks(conn, chunks, vectors, self.strategy)
                conn.commit()
                results.append(IngestResult(doc.doc_id, doc.title, status, len(chunks)))
        return results

    def delete(self, doc_id: str) -> bool:
        with get_conn() as conn:
            exists = repo.get_document_hash(conn, doc_id) is not None
            if exists:
                repo.delete_document(conn, doc_id)
                conn.commit()
        return exists