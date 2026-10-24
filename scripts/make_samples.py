import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document as DocxFile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

from app.config import settings

out = settings.data_dir / "samples"
out.mkdir(parents=True, exist_ok=True)

with (settings.data_dir / "hotpot" / "corpus.jsonl").open(encoding="utf-8") as f:
    rows = [json.loads(next(f)) for _ in range(3)]

# Markdown
r = rows[0]
(out / "sample.md").write_text(f"# {r['title']}\n\n{r['text']}\n", encoding="utf-8")

# DOCX
r = rows[1]
d = DocxFile()
d.add_heading(r["title"], level=1)
d.add_paragraph(r["text"])
d.save(out / "sample.docx")

# PDF
r = rows[2]
c = canvas.Canvas(str(out / "sample.pdf"), pagesize=A4)
c.setTitle(r["title"])
y = 800
for line in simpleSplit(r["text"], "Helvetica", 11, 500):
    c.drawString(50, y, line)
    y -= 15
c.save()

print("ფაილები შეიქმნა:", sorted(p.name for p in out.iterdir()))