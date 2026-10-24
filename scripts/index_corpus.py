"""კორპუსის ინდექსირება: parse -> dedup -> chunk -> embed -> ბაზა."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.connection import get_conn, init_schema
from app.db import repository as repo
from app.ingestion.chunking import CHUNKERS
from app.ingestion.dedup import Deduplicator
from app.ingestion.embedder import Embedder
from app.ingestion.parsers import parse_file

BATCH_DOCS = 2000


def main(strategy: str) -> None:
    init_schema()
    docs = parse_file(settings.data_dir / "hotpot" / "corpus.jsonl")
    result = Deduplicator().run(docs)
    docs = result.unique
    print(f"დოკუმენტები: {len(docs)} (ზუსტი დუბლ. ამოღებულია: {len(result.exact_duplicates)})")

    chunker = CHUNKERS[strategy]()
    embedder = Embedder()
    print("მოდელის მოწყობილობა:", embedder.model.device)

    with get_conn() as conn:
        for start in range(0, len(docs), BATCH_DOCS):
            batch = docs[start:start + BATCH_DOCS]
            chunks = []
            for d in batch:
                repo.upsert_document(conn, d)
                chunks.extend(chunker.chunk(d))
            # embedding-ს ვთვლით "სათაური + ტექსტი"-ზე, ტექსტს კი ისე ვინახავთ, როგორც არის
            vectors = embedder.embed_documents([f"{c.metadata['title']}. {c.text}" for c in chunks])
            repo.insert_chunks(conn, chunks, vectors, strategy)
            conn.commit()
            print(f"  {min(start + BATCH_DOCS, len(docs))}/{len(docs)}")
        print(repo.count_rows(conn))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "structural")