import hashlib
import html
import re
from dataclasses import dataclass, field


def normalize_text(text: str) -> str:
    """HTML entities-ს ვშლით, ჰარებს ვასუფთავებთ."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
    """ტექსტს ვასუფთავებთ ისე, რომ ერთნაირი შინაარსის ტექსტებს ერთი hash ჰქონდეთ."""
    return re.sub(r"\s+", " ", text).strip()
    

def compute_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass
class Document:
    doc_id: str                 # უნიკალური იდენტიფიკატორი (სათაური ან ფაილის სახელი)
    title: str
    text: str
    source: str                 # საიდან მოვიდა (ფაილის სახელი)
    format: str                 # jsonl / pdf / docx / md / txt
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.title = html.unescape(self.title)
        self.text = html.unescape(self.text)
        if not self.content_hash:
            self.content_hash = compute_hash(self.text)
        