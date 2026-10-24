import math
import re
from collections import Counter, defaultdict

from app.search.fusion import STOPWORDS, Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc_idx, tf)]
        self.lengths: list[int] = []
        self.meta: list[tuple[str, str, str, str]] = []   # chunk_id, doc_id, title, text
        self.avgdl = 0.0

    def build(self, rows: list[tuple[str, str, str, str]]) -> None:
        """rows: (chunk_id, doc_id, title, text). სათაურსაც ვაინდექსებთ."""
        for idx, row in enumerate(rows):
            tokens = tokenize(f"{row[2]} {row[3]}")
            self.lengths.append(len(tokens))
            self.meta.append(row)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        N = len(self.lengths)
        return math.log((N - n + 0.5) / (n + 0.5) + 1)

    def search(self, query: str, limit: int = 50) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for idx, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.lengths[idx] / self.avgdl
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [Hit(*self.meta[i], score=s) for i, s in top]

    @classmethod
    def from_db(cls, conn, strategy: str = "structural", **kw) -> "BM25Index":
        rows = conn.execute(
            """SELECT c.chunk_id, c.doc_id, d.title, c.text
               FROM chunks c JOIN documents d USING (doc_id)
               WHERE c.strategy = %s ORDER BY c.chunk_id""", (strategy,)
        ).fetchall()
        index = cls(**kw)
        index.build(rows)
        return index