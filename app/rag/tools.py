"""ინსტრუმენტები, რომლებსაც LLM იძახებს. ყველა გამოძახება მოწმდება და შეცდომები ტექსტად ბრუნდება."""
from app.db.connection import get_conn
from app.db.repository import get_document_text
from app.rag.pipeline import Source
from app.search.hybrid import HybridSearcher

TOOL_SPECS = """Available tools:
1. search_documents(query: string, top_k: int = 5)
   Hybrid search over the document collection. Returns numbered passages.
2. get_document(title: string)
   Returns the full text of one document by its exact title (use a title seen in search results).
"""

SNIPPET_CHARS = 700


class ToolBox:
    def __init__(self, searcher: HybridSearcher) -> None:
        self.searcher = searcher
        self.passages: dict[int, Source] = {}      # გლობალური ნუმერაცია მთელ საუბარზე
        self._by_chunk: dict[str, int] = {}

    def _register(self, chunk_id: str, doc_id: str, title: str, text: str) -> int:
        if chunk_id in self._by_chunk:
            return self._by_chunk[chunk_id]
        n = len(self.passages) + 1
        self.passages[n] = Source(n, chunk_id, doc_id, title, text)
        self._by_chunk[chunk_id] = n
        return n

    def search_documents(self, query: str, top_k: int = 5) -> str:
        top_k = max(1, min(int(top_k), 8))
        hits = self.searcher.search(str(query), top_k=top_k)
        if not hits:
            return "No results."
        lines = []
        for h in hits:
            n = self._register(h.chunk_id, h.doc_id, h.title, h.text)
            lines.append(f"[{n}] ({h.title}) {h.text[:SNIPPET_CHARS]}")
        return "\n".join(lines)

    def get_document(self, title: str) -> str:
        with get_conn() as conn:
            found = get_document_text(conn, str(title))
        if not found:
            return f"No document with title '{title}'."
        doc_id, real_title, text = found
        n = self._register(f"{doc_id}::full", doc_id, real_title, text)
        return f"[{n}] ({real_title}) {text[:3000]}"

    def call(self, name: str, args: dict) -> str:
        """ვალიდაცია: უცნობი ინსტრუმენტი ან არასწორი არგუმენტი ტექსტურ შეცდომად ბრუნდება."""
        if not isinstance(args, dict):
            return "ERROR: args must be a JSON object."
        try:
            if name == "search_documents":
                return self.search_documents(args["query"], args.get("top_k", 5))
            if name == "get_document":
                return self.get_document(args["title"])
        except KeyError as e:
            return f"ERROR: missing argument {e}."
        except Exception as e:                                  # noqa: BLE001
            return f"ERROR: tool failed ({e})."
        return f"ERROR: unknown tool '{name}'."