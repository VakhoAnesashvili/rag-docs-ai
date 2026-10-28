"""End-to-end eval: python eval/e2e_eval.py run <config> | report"""
import html
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.rag.agent import AgentPipeline
from app.rag.llm import LLMClient
from app.rag.pipeline import RagPipeline
from app.search.hybrid import HybridSearcher

OUT = Path("eval/results")
HOTPOT_OFFSET, HOTPOT_N = 300, 30

UNANSWERABLE = [
    "Who won the 2031 Nobel Prize in Physics?",
    "What is the phone number of the mayor of the city of Dwarka?",
    "Which spacecraft first landed humans on Mars?",
    "What was the exact salary of Vice-Admiral Iqbal Fazl Quadir in 2005?",
    "Who is the current CEO of the company that built the Type 346 radar?",
    "What was the winning lottery number in Norfolk on 1 January 2020?",
]


def norm(s: str) -> str:
    s = html.unescape(s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_questions() -> list[dict]:
    qs = []
    inc = settings.data_dir / "incidents" / "questions.jsonl"
    for line in inc.open(encoding="utf-8"):
        d = json.loads(line)
        d["source"] = "incident"
        qs.append(d)

    rows = [json.loads(l) for l in (settings.data_dir / "hotpot" / "questions.jsonl").open(encoding="utf-8")]
    random.Random(settings.random_seed).shuffle(rows)
    for r in rows[HOTPOT_OFFSET:HOTPOT_OFFSET + HOTPOT_N]:
        qs.append({"id": "hp_" + r["id"][:8], "question": r["question"], "answer": r["answer"],
                   "type": "hotpot_" + r["type"], "supporting_titles": r["supporting_titles"],
                   "expect": "answer", "source": "hotpot"})
    for i, q in enumerate(UNANSWERABLE, 1):
        qs.append({"id": f"un{i:02d}", "question": q, "answer": None, "type": "unanswerable",
                   "supporting_titles": [], "expect": "abstain", "source": "synthetic"})
    return qs


JUDGE_SYSTEM = """You grade an answer against a reference answer. Reply JSON only: {"correct": true|false}.
correct=true if the answer states the same fact(s) as the reference, even if longer or worded differently.
correct=false if it contradicts the reference, misses the key fact, or refuses to answer."""


def judge(llm: LLMClient, q: dict, answer: str) -> bool:
    prompt = f"QUESTION: {q['question']}\nREFERENCE: {q['answer']}\nANSWER: {answer}"
    try:
        return bool(llm.generate_json(prompt, system=JUDGE_SYSTEM).get("correct"))
    except Exception:                                  # noqa: BLE001
        return False


def build(config: str, searcher: HybridSearcher, llm: LLMClient):
    if config == "pipeline":
        return RagPipeline(searcher, llm)
    if config == "pipeline_nodecomp":
        return RagPipeline(searcher, llm, decompose=False)
    if config == "agent":
        return AgentPipeline(searcher, llm)
    raise SystemExit(f"unknown config {config}")


def run(config: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"e2e_{config}.jsonl"
    done = {json.loads(l)["id"] for l in path.open(encoding="utf-8")} if path.exists() else set()

    llm = LLMClient(order=("groq",))
    runner = build(config, HybridSearcher(), llm)
    questions = [q for q in load_questions() if q["id"] not in done]
    print(f"{config}: დარჩა {len(questions)} კითხვა (უკვე შესრულებული: {len(done)})")

    for i, q in enumerate(questions, 1):
        t0 = time.time()
        r = runner.ask(q["question"])
        gold = {norm(t) for t in q["supporting_titles"]}
        retrieved_titles = {norm(s.title) for s in r.retrieved}
        cited_titles = [norm(s.title) for s in r.sources]

        correct = None
        if q["expect"] == "answer" and r.answer:
            if q["source"] == "hotpot" and norm(q["answer"]) in norm(r.answer):
                correct = True
            else:
                correct = judge(llm, q, r.answer)

        rec = {
            "id": q["id"], "type": q["type"], "expect": q["expect"], "question": q["question"],
            "gold_answer": q["answer"], "answer": r.answer, "abstained": r.abstained,
            "conflict": r.conflict, "error": r.error, "correct": correct,
            "gold_titles": sorted(gold), "retrieved_titles": sorted(retrieved_titles),
            "cited_titles": cited_titles, "n_steps": len(r.trace), "sec": round(time.time() - t0, 1),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(questions)}] {q['id']} {q['type']} abstain={r.abstained} "
              f"correct={correct} {rec['sec']}s", flush=True)
        time.sleep(1)


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if not r["error"]]
    ans = [r for r in ok if r["expect"] == "answer"]
    gold_ans = [r for r in ans if r["gold_titles"]]
    answered = [r for r in ans if not r["abstained"] and r["cited_titles"]]
    return {
        "n": len(recs), "errors": len(recs) - len(ok),
        "answer_correctness": mean(1.0 if r["correct"] else 0.0 for r in ans),
        "recall_retrieved": mean(len(set(r["gold_titles"]) & set(r["retrieved_titles"])) / len(r["gold_titles"])
                                 for r in gold_ans),
        "both_found": mean(1.0 if set(r["gold_titles"]) <= set(r["retrieved_titles"]) else 0.0
                           for r in gold_ans),
        "citation_precision": mean(len(set(r["cited_titles"]) & set(r["gold_titles"])) / len(set(r["cited_titles"]))
                                   for r in answered if r["gold_titles"]),
        "citation_hit": mean(1.0 if set(r["cited_titles"]) & set(r["gold_titles"]) else 0.0
                             for r in answered if r["gold_titles"]),
        "abstention_acc": mean(1.0 if r["abstained"] else 0.0 for r in ok if r["expect"] == "abstain"),
        "conflict_detection": mean(1.0 if r["conflict"] else 0.0 for r in ok if r["expect"] == "conflict"),
        "avg_sec": mean(r["sec"] for r in recs),
    }


def report() -> None:
    table = {}
    for path in sorted(OUT.glob("e2e_*.jsonl")):
        recs = [json.loads(l) for l in path.open(encoding="utf-8")]
        name = path.stem.removeprefix("e2e_")
        table[name] = {"overall": summarize(recs)}
        by_type = defaultdict(list)
        for r in recs:
            by_type[r["type"]].append(r)
        table[name]["by_type"] = {t: summarize(v) for t, v in by_type.items()}
    (OUT / "e2e_summary.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, d in table.items():
        print("==", name, d["overall"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()