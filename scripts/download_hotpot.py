"""HotpotQA-დან ვაგებთ რეპროდუცირებად კორპუსს და შეფასების კითხვების ბაზას."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from app.config import settings


def main() -> None:
    out_dir = settings.data_dir / "hotpot"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"ჩამოტვირთულია {len(ds)} კითხვა")

    rng = random.Random(settings.random_seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    corpus: dict[str, dict] = {}  # title -> ჩანაწერი (დუბლიკატები title-ით იკარგება)
    questions: list[dict] = []

    for idx in indices:
        row = ds[idx]
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            if title not in corpus:
                corpus[title] = {"title": title, "text": " ".join(sents)}

        questions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
            }
        )
        if len(corpus) >= settings.corpus_size:
            break

    (out_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in corpus.values()),
        encoding="utf-8",
    )
    (out_dir / "questions.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questions),
        encoding="utf-8",
    )
    print(f"კორპუსი: {len(corpus)} პარაგრაფი | კითხვები: {len(questions)}")


if __name__ == "__main__":
    main()