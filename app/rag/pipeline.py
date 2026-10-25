import logging
from dataclasses import dataclass, field

from app.rag import prompts
from app.rag.llm import LLMClient, LLMError
from app.search.fusion import Hit
from app.search.hybrid import HybridSearcher

log = logging.getLogger(__name__)


@dataclass
class Source:
    number: int
    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass
class RagAnswer:
    question: str
    answer: str | None
    abstained: bool
    conflict: bool = False
    conflict_note: str | None = None
    sources: list[Source] = field(default_factory=list)       # მხოლოდ ციტირებული
    retrieved: list[Source] = field(default_factory=list)     # LLM-ს რაც მივეცით
    sub_queries: list[str] = field(default_factory=list)
    provider: str = ""
    error: str | None = None


class RagPipeline:
    def __init__(self, searcher: HybridSearcher, llm: LLMClient,
                 per_query_k: int = 6, max_passages: int = 10, decompose: bool = True) -> None:
        self.searcher, self.llm = searcher, llm
        self.per_query_k, self.max_passages = per_query_k, max_passages
        self.decompose = decompose

    # 1. DECOMPOSE
    def _decompose(self, question: str) -> list[str]:
        if not self.decompose:
            return [question]
        try:
            data = self.llm.generate_json(question, system=prompts.DECOMPOSE_SYSTEM)
            queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
            return queries[:3] or [question]
        except LLMError as e:
            log.warning("decompose ჩავარდა, ვიყენებთ ორიგინალ კითხვას: %s", e)
            return [question]

    # 2. RETRIEVE
    def _retrieve(self, question: str, sub_queries: list[str]) -> list[Hit]:
        # ორიგინალი კითხვაც ყოველთვის შედის
        queries = [question] + [q for q in sub_queries if q != question]
        merged: dict[str, Hit] = {}
        for q in queries:
            for h in self.searcher.search(q, top_k=self.per_query_k):
                if h.chunk_id not in merged:
                    merged[h.chunk_id] = h
        return list(merged.values())[: self.max_passages]

    # 3 + 4. ANSWER და VERIFY
    def ask(self, question: str) -> RagAnswer:
        sub_queries = self._decompose(question)
        hits = self._retrieve(question, sub_queries)
        retrieved = [Source(i, h.chunk_id, h.doc_id, h.title, h.text)
                     for i, h in enumerate(hits, 1)]
        result = RagAnswer(question, None, True, sub_queries=sub_queries, retrieved=retrieved)
        if not hits:
            return result

        prompt = prompts.build_answer_prompt(question, [(s.title, s.text) for s in retrieved])
        try:
            data = self.llm.generate_json(prompt, system=prompts.ANSWER_SYSTEM)
        except LLMError as e:
            result.error = str(e)
            return result

        by_number = {s.number: s for s in retrieved}
        cited = [n for n in data.get("citations", []) if isinstance(n, int)]
        valid = [n for n in dict.fromkeys(cited) if n in by_number]   # VERIFY: გამოგონილი ნომრები ვარდება

        sufficient = bool(data.get("sufficient")) and data.get("answer")
        if sufficient and not valid:
            sufficient = False          # პასუხი წყაროს გარეშე არ ვიღებთ

        result.abstained = not sufficient
        result.answer = data.get("answer") if sufficient else None
        result.conflict = bool(data.get("conflict"))
        result.conflict_note = data.get("conflict_note")
        result.sources = [by_number[n] for n in valid]
        return result