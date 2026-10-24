from app.db.connection import get_conn
from app.ingestion.embedder import Embedder
from app.search.bm25 import BM25Index
from app.search.fusion import Hit, keyword_terms, rrf_fuse, title_boost

_DENSE_SQL = """
SELECT c.chunk_id, c.doc_id, d.title, c.text
FROM chunks c JOIN documents d USING (doc_id)
WHERE c.strategy = %s
ORDER BY c.embedding <=> %s
LIMIT %s
"""

_SPARSE_SQL = """
SELECT c.chunk_id, c.doc_id, d.title, c.text
FROM chunks c JOIN documents d USING (doc_id)
WHERE c.strategy = %s AND c.tsv @@ to_tsquery('english', %s)
ORDER BY ts_rank_cd(c.tsv, to_tsquery('english', %s)) DESC
LIMIT %s
"""


class HybridSearcher:
    def __init__(self, embedder: Embedder | None = None, strategy: str = "structural") -> None:
        self.embedder = embedder or Embedder()
        self.strategy = strategy
        self._bm25 = None
    
    def dense(self, conn, query: str, limit: int) -> list[Hit]:
        vec = self.embedder.embed_query(query)
        rows = conn.execute(_DENSE_SQL, (self.strategy, vec, limit)).fetchall()
        return [Hit(*r) for r in rows]

    def sparse(self, conn, query: str, limit: int) -> list[Hit]:
        terms = keyword_terms(query)
        if not terms:
            return []
        tsq = " | ".join(terms)
        rows = conn.execute(_SPARSE_SQL, (self.strategy, tsq, tsq, limit)).fetchall()
        return [Hit(*r) for r in rows]
    
    def bm25(self, conn, query: str, limit: int):
            if self._bm25 is None:                       # ერთხელ ვაგებთ, მერე მეხსიერებაში რჩება
                self._bm25 = BM25Index.from_db(conn, self.strategy)
            return self._bm25.search(query, limit)

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid",
               candidates: int = 50, rerank: bool = True) -> list[Hit]:
        with get_conn() as conn:
            if mode == "dense":
                return self.dense(conn, query, top_k)
            if mode == "sparse_pg":
                return self.sparse_pg(conn, query, top_k)
            if mode == "sparse":
                return self.bm25(conn, query, top_k)
            rankings = {
                "dense": self.dense(conn, query, candidates),
                "sparse": self.bm25(conn, query, candidates),
            }
        fused = rrf_fuse(rankings)
        if rerank:
            fused = title_boost(fused, query)
        return fused[:top_k]