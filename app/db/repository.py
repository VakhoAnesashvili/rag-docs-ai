import json

import numpy as np

from app.db.connection import get_conn
from app.ingestion.chunking import Chunk
from app.ingestion.models import Document


def upsert_document(conn, doc: Document) -> None:
    conn.execute(
        """INSERT INTO documents (doc_id, title, source, format, content_hash, metadata)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (doc_id) DO UPDATE SET
             title = EXCLUDED.title, source = EXCLUDED.source,
             format = EXCLUDED.format, content_hash = EXCLUDED.content_hash,
             metadata = EXCLUDED.metadata""",
        (doc.doc_id, doc.title, doc.source, doc.format, doc.content_hash,
         json.dumps(doc.metadata)),
    )


def insert_chunks(conn, chunks: list[Chunk], embeddings: np.ndarray, strategy: str) -> None:
    rows = [
        (c.chunk_id, strategy, c.doc_id, c.index, c.text, emb)
        for c, emb in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (chunk_id, strategy, doc_id, chunk_index, text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (chunk_id, strategy) DO NOTHING""",
            rows,
        )


def delete_document(conn, doc_id: str) -> None:
    conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))   # cascade


def get_document_hash(conn, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def count_rows(conn) -> dict:
    return {
        "documents": conn.execute("SELECT count(*) FROM documents").fetchone()[0],
        "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
    }

def get_document_text(conn, key: str, strategy: str = "structural") -> tuple[str, str, str] | None:
    """key = doc_id ან სათაური. აბრუნებს (doc_id, title, სრული ტექსტი)."""
    rows = conn.execute(
        """SELECT c.doc_id, d.title, c.text
           FROM chunks c JOIN documents d USING (doc_id)
           WHERE c.strategy = %s AND (c.doc_id = %s OR lower(d.title) = lower(%s))
           ORDER BY c.chunk_index""",
        (strategy, key, key),
    ).fetchall()
    if not rows:
        return None
    return rows[0][0], rows[0][1], " ".join(r[2] for r in rows)