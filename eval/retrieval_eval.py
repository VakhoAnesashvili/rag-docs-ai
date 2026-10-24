"""Retrieval-ის შედარება: dense / sparse_pg / sparse (BM25) / hybrid."""
import html
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.search.hybrid import HybridSearcher
from eval.metrics import both_at_k, recall_at_k, reciprocal_rank

KS = (5, 10, 20)
N_QUESTIONS = 300


def norm(title: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(title)).strip().lower()


def load_questions(n: int) -> list[dict]:
    path = settings.data_dir / "hotpot" / "questions.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    random.Random(settings.random_seed).shuffle(rows)
    return rows[:n]


def evaluate(searcher: HybridSearcher, questions: list[dict], mode: str, rerank: bool = True) -> dict:
    sums = {f"recall@{k}": 0.0 for k in KS} | {f"both@{k}": 0.0 for k in KS} | {"mrr": 0.0}
    t0 = time.time()
    for q in questions:
        gold = {norm(t) for t in q["supporting_titles"]}
        hits = searcher.search(q["question"], top_k=max(KS), mode=mode, rerank=rerank)
        titles = []
        for h in hits:                      # chunk-ებიდან უნიკალური სათაურები, რიგის შენარჩუნებით
            t = norm(h.title)
            if t not in titles:
                titles.append(t)
        for k in KS:
            sums[f"recall@{k}"] += recall_at_k(titles, gold, k)
            sums[f"both@{k}"] += both_at_k(titles, gold, k)
        sums["mrr"] += reciprocal_rank(titles, gold)
    n = len(questions)
    out = {k: round(v / n, 3) for k, v in sums.items()}
    out["sec/q"] = round((time.time() - t0) / n, 3)
    return out


def main() -> None:
    questions = load_questions(N_QUESTIONS)
    searcher = HybridSearcher()
    configs = [
        ("dense", "dense", False),
        ("sparse_pg", "sparse_pg", False),
        ("bm25", "sparse", False),
        ("hybrid (RRF)", "hybrid", False),
        ("hybrid + title boost", "hybrid", True),
    ]
    results = {}
    for name, mode, rerank in configs:
        print(f"== {name}", flush=True)
        results[name] = evaluate(searcher, questions, mode, rerank)
        print(results[name], flush=True)

    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()