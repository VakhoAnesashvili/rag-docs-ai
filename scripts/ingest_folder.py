import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import init_schema
from app.ingestion.embedder import Embedder
from app.ingestion.service import IngestionService

folder = Path(sys.argv[1])
init_schema()
svc = IngestionService(Embedder())
for p in sorted(folder.iterdir()):
    if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
        for r in svc.ingest_file(p):
            print(f"{p.name}: {r.status} | title={r.title!r} | chunks={r.chunks}")