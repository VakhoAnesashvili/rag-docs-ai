import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ingestion.models import Document


@dataclass
class Chunk:
    chunk_id: str               # "<doc_id>::<ინდექსი>"
    doc_id: str
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    def _make(self, doc: Document, index: int, text: str, **extra) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}::{index}",
            doc_id=doc.doc_id,
            text=text,
            index=index,
            metadata={"title": doc.title, "source": doc.source,
                      "format": doc.format, **extra},
        )


class FixedChunker(BaseChunker):
    """სტრატეგია A: ფიქსირებული სიტყვების რაოდენობა + overlap."""
    name = "fixed"

    def __init__(self, size: int = 200, overlap: int = 40) -> None:
        if overlap >= size:
            raise ValueError("overlap უნდა იყოს size-ზე ნაკლები")
        self.size, self.overlap = size, overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        if not words:
            return []
        step = self.size - self.overlap
        chunks, i = [], 0
        while i < len(words):
            piece = words[i:i + self.size]
            chunks.append(self._make(doc, len(chunks), " ".join(piece)))
            if i + self.size >= len(words):
                break
            i += step
        return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class StructuralChunker(BaseChunker):
    """სტრატეგია B: სათაურები -> აბზაცები -> წინადადებები."""
    name = "structural"

    def __init__(self, max_words: int = 200) -> None:
        self.max_words = max_words

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_title, body in self._sections(doc):
            for piece in self._pack(body):
                chunks.append(self._make(doc, len(chunks), piece,
                                         section=section_title))
        return chunks

    @staticmethod
    def _sections(doc: Document) -> list[tuple[str, str]]:
        """ტექსტს ვყოფთ '# სათაური' ხაზებით."""
        sections, current_title, buf = [], doc.title, []
        for line in doc.text.splitlines():
            if line.startswith("#"):
                if "".join(buf).strip():
                    sections.append((current_title, "\n".join(buf)))
                current_title, buf = line.lstrip("# ").strip(), []
            else:
                buf.append(line)
        if "".join(buf).strip():
            sections.append((current_title, "\n".join(buf)))
        return sections

    def _pack(self, body: str) -> list[str]:
        """აბზაცებს ვაერთიანებთ max_words-მდე; გრძელს წინადადებებით ვჭრით."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        units: list[str] = []
        for p in paragraphs:
            if len(p.split()) <= self.max_words:
                units.append(p)
            else:
                units.extend(s for s in _SENTENCE_SPLIT.split(p) if s)

        packed, current, count = [], [], 0
        for u in units:
            n = len(u.split())
            if current and count + n > self.max_words:
                packed.append(" ".join(current))
                current, count = [], 0
            current.append(u)
            count += n
        if current:
            packed.append(" ".join(current))
        return packed


CHUNKERS = {"fixed": FixedChunker, "structural": StructuralChunker} 